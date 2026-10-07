#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

WORKDIR = Path("./work_tx_address_gds13_train_centrality")
OUTPUT_DIR = WORKDIR / "profile_lift_analysis"
TRAIN_END = "2021-05-17"
TEST_START = "2021-05-18"
RULE_VERSION = "behavior_profiles_v1"
PROFILES = ("T1", "T2", "T3", "T4")
NAMES = {
    "T1": "High Transaction Density",
    "T2": "High-Value Transfer",
    "T3": "High Sender Participation",
    "T4": "High-Centrality Address Involvement",
}
METRICS = ("degree", "pageRank", "eigenvector")
FEATURES = ["value", "tx_count_15min_z", "active_sender_15m_log"] + [
    f"{side}_{metric}_log"
    for side in ("sender", "receiver") for metric in METRICS
]


def load_thresholds(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("rule_version") != RULE_VERSION:
        raise ValueError("Profile rule version does not match behavior_profiles_v1")
    if data.get("train_end_exclusive") != TRAIN_END:
        raise ValueError("Profile TRAIN boundary mismatch")
    if data.get("baseline_quantile") != .95:
        raise ValueError("This analysis expects the baseline P95 profiles")
    if data.get("value_scale") != "stored value column (log1p DOGE in this pipeline)":
        raise ValueError("Unrecognized value scale; do not compare raw DOGE with log thresholds")
    thresholds = data["thresholds"]
    for key in ("density", "value", "senders", *METRICS):
        if not np.isfinite(float(thresholds[key])):
            raise ValueError(f"Non-finite threshold: {key}")
    return thresholds, data, hashlib.sha256(raw).hexdigest()


def behavior_profile_flags(df, thresholds):

    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing profile features: {missing}")
    x = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Missing/non-finite profile features")
    flags = pd.DataFrame(index=df.index)
    flags["T1"] = (x["tx_count_15min_z"] >= thresholds["density"]) & (x["tx_count_15min_z"] > 0)
    flags["T2"] = (x["value"] >= thresholds["value"]) & (x["value"] > 0)
    flags["T3"] = (x["active_sender_15m_log"] >= thresholds["senders"]) & (x["active_sender_15m_log"] > 0)
    endpoint_high = []
    for side in ("sender", "receiver"):
        votes = sum(
            ((x[f"{side}_{metric}_log"] >= thresholds[metric])
             & (x[f"{side}_{metric}_log"] > 0)).astype("int8")
            for metric in METRICS
        )
        endpoint_high.append(votes >= 2)
    flags["T4"] = endpoint_high[0] | endpoint_high[1]
    return flags


def aggregate_chunk(df, thresholds):
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("Invalid dates; records must not be silently dropped")
    labels = pd.to_numeric(df["final_anomaly"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("final_anomaly must contain only 0 or 1")
    flags = behavior_profile_flags(df, thresholds)
    daily = flags.astype("int64")
    daily["day"] = dates.dt.strftime("%Y-%m-%d")
    daily["label"] = labels.astype("int8")
    daily["n"] = 1
    return daily.groupby(["day", "label"], as_index=False)[["n", *PROFILES]].sum()


def split_from_day(day):
    return "TRAIN" if day < TRAIN_END else ("VALID" if day < TEST_START else "TEST")


def summarize(counts, scope):
    sums = counts.groupby("label")[["n", *PROFILES]].sum()

    sums = sums.reindex([0, 1], fill_value=0)
    normal_n, anomaly_n = int(sums.loc[0, "n"]), int(sums.loc[1, "n"])
    rows = []
    for tid in PROFILES:
        normal_match = int(sums.loc[0, tid])
        anomaly_match = int(sums.loc[1, tid])
        pn = normal_match / normal_n if normal_n else np.nan
        pa = anomaly_match / anomaly_n if anomaly_n else np.nan
        if not normal_n or not anomaly_n:
            lift, status = np.nan, "undefined_empty_label_group"
        elif normal_match == 0:
            lift = np.inf if anomaly_match else np.nan
            status = "infinite_no_normal_matches" if anomaly_match else "undefined_no_matches"
        else:
            lift, status = pa / pn, "finite"
        rows.append({
            "scope": scope, "profile": tid, "profile_name": NAMES[tid],
            "anomaly_total": anomaly_n, "normal_total": normal_n,
            "anomaly_profile_count": anomaly_match, "normal_profile_count": normal_match,
            "anomaly_profile_rate_percent": pa * 100,
            "normal_profile_rate_percent": pn * 100,
            "difference_percentage_points": (pa - pn) * 100,
            "lift": lift, "lift_status": status,
        })
    return rows


def build_tables(daily):
    daily = daily.groupby(["day", "label"], as_index=False)[["n", *PROFILES]].sum()
    daily["dataset_split"] = daily["day"].map(split_from_day)
    overall = pd.DataFrame(summarize(daily, "ALL"))
    by_split = pd.DataFrame([
        row for split in ("TRAIN", "VALID", "TEST")
        for row in summarize(daily.loc[daily["dataset_split"].eq(split)], split)
    ])
    by_day = pd.DataFrame([
        row for day, group in daily.groupby("day", sort=True)
        for row in summarize(group, day)
    ])
    by_day["dataset_split"] = by_day["scope"].map(split_from_day)
    return daily, overall, by_split, by_day


def format_number(value):
    if pd.isna(value):
        return "--"
    if np.isinf(value):
        return r"$\infty$"
    return f"{value:.3f}"


def latex_table(table):
    lines = [r"\begin{table}[!t]", r"\caption{Profile Prevalence and Lift in the Test Period}",
             r"\label{tab:profile_lift_test}", r"\centering", r"\scriptsize",
             r"\begin{tabular}{lrrrr}", r"\hline",
             r"Profile & Anomaly (\%) & NORMAL (\%) & Difference (pp) & Lift \\",
             r"\hline"]
    for _, row in table.iterrows():
        numbers = [row[c] for c in ("anomaly_profile_rate_percent",
                   "normal_profile_rate_percent", "difference_percentage_points", "lift")]
        lines.append(row["profile"] + " & " + " & ".join(map(format_number, numbers)) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\vspace{2pt}",
              r"\begin{minipage}{\columnwidth}\scriptsize",
              "Profiles are evaluated independently using fixed TRAIN-derived thresholds. "
              "T4 components retained within T5 are included. NORMAL denotes records "
              "not selected by Isolation Forest. Lift is the anomaly-group profile "
              "prevalence divided by the NORMAL-group prevalence. "
              "These are descriptive record-level comparisons, not independent ground-truth validation.",
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    threshold_path = WORKDIR / "profile_thresholds.json"
    thresholds, metadata, digest = load_thresholds(threshold_path)
    files = sorted(WORKDIR.glob("base_scored_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No base_scored_chunk_*.parquet in {WORKDIR.resolve()}. "
                                "All scored records, including label 0, are required.")
    parts, manifest = [], []
    columns = ["date", "final_anomaly", *FEATURES]
    for index, path in enumerate(files, 1):

        df = pd.read_parquet(path, columns=columns)
        try:
            counts = aggregate_chunk(df, thresholds)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        parts.append(counts)
        manifest.append({"name": path.name, "rows": len(df),
                         "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
        print(f"[{index}/{len(files)}] {path.name}: {len(df):,} records", flush=True)
        del df
    daily, overall, by_split, by_day = build_tables(pd.concat(parts, ignore_index=True))
    if not daily.loc[daily["label"].eq(0), "n"].sum():
        raise ValueError("No NORMAL records. Anomaly-only exports cannot be used for lift.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, table in (
        ("profile_lift_overall.csv", overall), ("profile_lift_by_split.csv", by_split),
        ("profile_lift_daily.csv", by_day), ("profile_lift_daily_counts.csv", daily),
    ):
        table.to_csv(OUTPUT_DIR / filename, index=False, float_format="%.8f", na_rep="NA")
    test_table = by_split.loc[by_split["scope"].eq("TEST")]
    (OUTPUT_DIR / "profile_lift_test_table.tex").write_text(latex_table(test_table), encoding="utf-8")
    notes = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "unit": "derived input-output association record; not unique blockchain transaction",
        "comparison": "P(profile|final_anomaly=1) / P(profile|final_anomaly=0)",
        "rule_version": RULE_VERSION, "thresholds": thresholds,
        "threshold_file_sha256": digest, "threshold_metadata": metadata,
        "train_end_exclusive": TRAIN_END, "test_start_inclusive": TEST_START,
        "normal_definition": "IF label 0; not independently verified normal",
        "source_directory": str(WORKDIR.resolve()), "input_manifest": manifest,
        "total_records": int(daily["n"].sum()),
        "anomaly_records": int(daily.loc[daily["label"].eq(1), "n"].sum()),
        "interpretation": [
            "All four profile conditions are reevaluated for BOTH label groups.",
            "All overlaps, including T4 within T5, contribute to component prevalence.",
            "Rates are record-weighted, including repeated window-level features.",
            "No smoothing or artificial pseudocounts are used; NA/inf are explicit.",
            "Only base_scored_chunk files are read to avoid counting scored copies twice.",
            "Each input family must represent one complete run without overlapping records.",
            "Lift describes association with detector selection; it is not accuracy or causality.",
            "No row-independent p-values or confidence intervals are claimed: records share "
            "timestamps/windows and may share source transactions.",
            "ALL includes in-sample TRAIN records; TEST is the main held-out comparison.",
        ],
    }
    (OUTPUT_DIR / "profile_lift_analysis_notes.json").write_text(
        json.dumps(notes, indent=2, allow_nan=False), encoding="utf-8")
    print("\nTEST results:\n" + test_table.to_string(index=False))
    print(f"\nSaved outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
