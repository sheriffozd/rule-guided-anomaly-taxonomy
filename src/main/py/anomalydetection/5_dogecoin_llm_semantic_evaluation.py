import os
import re
import json
import time
import pandas as pd
from openai import OpenAI

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score


WORKDIR = "./resources/work_tx_address_gds13_train_centrality"
EVENT_CALENDAR_PATH = os.path.join(WORKDIR, "event_calendar.csv")

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

PROFILE_COMPONENTS = {"T1", "T2", "T3", "T4"}
ALLOWED_TAXONOMIES = PROFILE_COMPONENTS

TAXONOMY_NAMES = {
    "T1": "High Transaction Density",
    "T2": "High-Value Transfer",
    "T3": "High Sender Participation",
    "T4": "High-Centrality Address Involvement",
    "T5": "Composite Profile",
    "T6": "Uncharacterized Anomaly",
    "NORMAL": "Normal",
}

TAXONOMY_DEFINITIONS = """
T1: High Transaction Density. A transaction occurs in a 15-minute window whose
transaction-count z-score reaches the TRAIN q95 cutoff and is positive.

T2: High-Value Transfer. The transaction's stored log1p transfer value reaches
the TRAIN q95 cutoff and is positive.

T3: High Sender Participation. The number of active senders in the transaction's
15-minute window reaches the TRAIN q95 cutoff and is positive.

T4: High-Centrality Address Involvement. The sender or receiver, considered as
one endpoint, reaches TRAIN q95 in at least two of degree, PageRank, and
eigenvector centrality.

T5 is derived when at least two T1--T4 components are present. T6 is derived
when none is present. T5 and T6 are not independent event-level components.
"""


def load_event_calendar(path):
    df = pd.read_csv(path)

    df["event_datetime"] = pd.to_datetime(
        df["event_date"].astype(str) + " " + df["event_time"].astype(str),
        errors="coerce"
    )

    df["human_expected_taxonomy"] = (
        df["expected_taxonomy"]
        .fillna("")
        .astype(str)
        .str.replace("&", ",", regex=False)
        .str.replace(" ", "", regex=False)
    )

    return df


def extract_json(text):
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text}")

    return json.loads(match.group(0))


def build_llm_prompt(row):
    return f"""
You are an independent auxiliary semantic evaluator for Dogecoin anomaly taxonomy validation.

Your task is to identify which anomaly taxonomy categories could plausibly
be associated with the transaction behavior expected around the documented event.

Taxonomy definitions:
{TAXONOMY_DEFINITIONS}

Event information:
event_id = {row.get("event_id", "")}
event_name = {row.get("event_name", "")}
event_type = {row.get("event_type", "")}
event_date = {row.get("event_date", "")}
event_time = {row.get("event_time", "")}
window_hours = {row.get("window_hours", "")}

Decision rules:
Choose 1 or 2 component labels from T1,T2,T3,T4.

Base the prediction only on the taxonomy definitions and event information provided.

Do not assume or infer any human-assigned taxonomy label.

The task is not to determine whether the event proves that an anomaly occurred. Instead, identify which transaction-level anomaly behaviors would be semantically plausible around the event.

Do not require direct transaction evidence in the event description.

Consider the type and timing of the event when assessing plausible short-term transaction responses.

Select T1 for plausible unusually high short-window transaction density.

Select T2 for plausible unusually high individual transfer values.

Select T3 for plausible unusually broad sender participation.

Select T4 for plausible involvement of structurally central addresses.

Do not output T5 or T6; they are derived from the number of observed components.

Select at most two labels with the strongest semantic justification.

primary_taxonomy must be one of ai_expected_taxonomy.


Return ONLY valid JSON:
{{
  "event_id": "{row.get("event_id", "")}",
  "ai_expected_taxonomy": ["T1"],
  "primary_taxonomy": "T1",
  "confidence": 0.5,
  "short_reason": "brief semantic justification based only on the event and taxonomy definitions"
}}
"""


def validate_llm_result(data, row):
    event_id = str(row.get("event_id", ""))

    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object.")

    tax = data.get("ai_expected_taxonomy", [])

    if isinstance(tax, str):
        tax = [tax]

    tax = [
        str(t).strip().upper()
        for t in tax
        if str(t).strip().upper() in ALLOWED_TAXONOMIES
    ]

    tax = tax[:2]

    primary = str(data.get("primary_taxonomy", "")).strip().upper()

    if primary not in ALLOWED_TAXONOMIES:
        primary = tax[0] if tax else "T1"

    if primary not in tax:
        tax = [primary] + [t for t in tax if t != primary]

    tax = tax[:2]

    if not tax:
        tax = ["T1"]
        primary = "T1"

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("short_reason", "")).strip()
    if not reason:
        reason = (
            "LLM returned limited explanation; assigned using fallback validation."
        )

    return {
        "event_id": str(data.get("event_id") or event_id),
        "ai_expected_taxonomy": tax,
        "primary_taxonomy": primary,
        "confidence": confidence,
        "short_reason": reason
    }


def call_llm_auxiliary_evaluator(row, model="qwen2.5:7b", max_retries=3):
    prompt = build_llm_prompt(row)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY one valid JSON object. "
                            "Do not use markdown. "
                            "Do not explain outside JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            text = response.choices[0].message.content.strip()
            data = extract_json(text)
            return validate_llm_result(data, row)

        except Exception as e:
            print(
                f"[WARN] LLM failed for "
                f"{row.get('event_id', 'UNKNOWN')} "
                f"attempt={attempt + 1}: {e}"
            )
            time.sleep(2)

    return {
        "event_id": str(row.get("event_id", "")),
        "ai_expected_taxonomy": ["T1"],
        "primary_taxonomy": "T1",
        "confidence": 0.0,
        "short_reason": (
            "LLM evaluation failed; deterministic T1 fallback used for an auditable missing-result marker."
        )
    }


def event_jaccard_score(expected, predicted):
    if pd.isna(expected) or predicted is None:
        return 0.0

    expected_set = {
        x for x in str(expected).replace(" ", "").split(",")
        if x in ALLOWED_TAXONOMIES
    }

    predicted_set = {
        x for x in predicted
        if x in ALLOWED_TAXONOMIES
    }

    union = expected_set | predicted_set

    if not union:
        return 0.0

    return round(
        len(expected_set & predicted_set) / len(union),
        3
    )


def exact_or_partial(expected, predicted):
    expected_set = {
        x for x in str(expected).replace(" ", "").split(",")
        if x in ALLOWED_TAXONOMIES
    }

    predicted_set = {
        x for x in predicted
        if x in ALLOWED_TAXONOMIES
    }

    if expected_set == predicted_set:
        return "exact"
    elif len(expected_set & predicted_set) > 0:
        return "partial"
    else:
        return "none"


def agreement_metrics(result_df):


    human_primary = []
    llm_primary = []

    for _, r in result_df.iterrows():
        h = str(r["human_expected_taxonomy"]).split(",")[0]
        l = str(r["ai_primary_taxonomy"])

        human_primary.append(h)
        llm_primary.append(l)

    kappa = cohen_kappa_score(human_primary, llm_primary)

    human_multi = [
        [x for x in str(v).split(",") if x in ALLOWED_TAXONOMIES]
        for v in result_df["human_expected_taxonomy"]
    ]

    llm_multi = [
        [x for x in str(v).split(",") if x in ALLOWED_TAXONOMIES]
        for v in result_df["ai_expected_taxonomy"]
    ]

    mlb = MultiLabelBinarizer(
        classes=["T1", "T2", "T3", "T4"]
    )

    Y_true = mlb.fit_transform(human_multi)
    Y_pred = mlb.transform(llm_multi)

    jaccard = jaccard_score(
        Y_true,
        Y_pred,
        average="samples",
        zero_division=0
    )

    f1 = f1_score(
        Y_true,
        Y_pred,
        average="micro",
        zero_division=0
    )

    print("\n========== AGREEMENT METRICS ==========")
    print(f"Cohen Kappa : {kappa:.4f}")
    print(f"Jaccard     : {jaccard:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print("=======================================\n")

    return {
        "cohen_kappa": kappa,
        "jaccard_samples": jaccard,
        "f1_micro": f1
    }


def run_llm_auxiliary_semantic_validation(
    event_calendar_path=EVENT_CALENDAR_PATH,
    model="qwen2.5:7b"
):
    events = load_event_calendar(event_calendar_path)

    rows = []

    for _, row in events.iterrows():
        print(
            f"[INFO] Evaluating "
            f"{row['event_id']} - {row['event_name']}"
        )

        llm_result = call_llm_auxiliary_evaluator(
            row,
            model=model
        )

        ai_labels = llm_result["ai_expected_taxonomy"]

        rows.append({
            "event_id": row["event_id"],
            "event_name": row["event_name"],
            "event_type": row.get("event_type", ""),
            "event_datetime": row["event_datetime"],
            "human_expected_taxonomy": row["human_expected_taxonomy"],
            "ai_expected_taxonomy": ",".join(ai_labels),
            "ai_derived_final_taxonomy": ai_labels[0] if len(ai_labels) == 1 else "T5",
            "ai_primary_taxonomy": llm_result["primary_taxonomy"],
            "ai_confidence": llm_result["confidence"],
            "human_ai_jaccard": event_jaccard_score(
                row["human_expected_taxonomy"],
                ai_labels
            ),
            "human_ai_agreement": exact_or_partial(
                row["human_expected_taxonomy"],
                ai_labels
            ),
            "ai_short_reason": llm_result["short_reason"]
        })

    result_df = pd.DataFrame(rows)

    metrics = agreement_metrics(result_df)

    out_path = os.path.join(
        WORKDIR,
        "llm_auxiliary_semantic_validation.csv"
    )
    result_df.to_csv(out_path, index=False)

    metrics_path = os.path.join(
        WORKDIR,
        "llm_auxiliary_semantic_metrics.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: float(v) for k, v in metrics.items()},
            f,
            indent=2
        )

    print("[OK] Saved:", out_path)
    print("[OK] Saved:", metrics_path)
    print(result_df)

    return result_df


def main():
    run_llm_auxiliary_semantic_validation(
        event_calendar_path=EVENT_CALENDAR_PATH,
        model="qwen2.5:7b"
    )


if __name__ == "__main__":
    main()