#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_WORKDIR = "./resources/work_tx_address_gds13_train_centrality"
DEFAULT_OUTPUT_SUBDIR = "taxonomy_match_null_results"
DEFAULT_PARQUET_PATTERN = "scored_chunk_*.parquet"
DEFAULT_N_PERMUTATIONS = 10000
DEFAULT_RANDOM_STATE = 42

DATE_COL = "date"
ANOMALY_COL = "final_anomaly"
TAXONOMY_COL = "taxonomy_id"
COMPONENT_COL = "taxonomy_components"

VALID_TAXONOMIES = ("T1", "T2", "T3", "T4", "T5", "T6")
EXPECTED_COMPONENTS = ("T1", "T2", "T3", "T4")


def read_csv_safely(path: Path) -> pd.DataFrame:
    last_error = None

    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            kwargs = {"low_memory": False}
            if enc is not None:
                kwargs["encoding"] = enc
            return pd.read_csv(path, **kwargs)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read CSV: {path}\n{last_error}")


def detect_event_file(
    workdir: Path,
    explicit: Optional[str]
) -> Path:

    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = workdir / p
        p = p.resolve()

        if not p.exists():
            raise FileNotFoundError(f"Event file not found: {p}")

        return p

    candidates = sorted(
        p for p in workdir.glob("event_calendar*.csv")
        if p.is_file()
    )

    if not candidates:
        raise FileNotFoundError(
            f"No event_calendar*.csv found under {workdir}"
        )

    return candidates[0]


def detect_scored_parquets(
    workdir: Path,
    pattern: str
) -> list[Path]:

    files = sorted(
        p for p in workdir.glob(pattern)
        if p.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No files matched {workdir / pattern}\n"
            "Expected scored_chunk_*.parquet files."
        )

    return files


def parse_expected_taxonomy(value) -> tuple[str, ...]:
    if pd.isna(value):
        return tuple()

    text = (
        str(value)
        .replace("&", ",")
        .replace(";", ",")
        .replace(" ", "")
        .strip()
    )

    values = [
        x for x in text.split(",")
        if x in EXPECTED_COMPONENTS
    ]


    return tuple(dict.fromkeys(values))


def load_event_calendar(path: Path) -> pd.DataFrame:
    events = read_csv_safely(path)

    required = {
        "event_id",
        "event_date",
        "event_time",
        "expected_taxonomy",
        "window_hours",
    }

    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(events.columns)}"
        )

    out = events.copy()

    out["_event_date"] = pd.to_datetime(
        out["event_date"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()

    time_text = out["event_time"].astype("string").str.strip()
    time_missing = time_text.isna() | time_text.eq("")

    if "time_precision" in out.columns:
        precision_text = (
            out["time_precision"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        explicit_day = precision_text.isin(
            {"day", "date", "all_day", "all-day", "daily"}
        )
    else:
        explicit_day = pd.Series(False, index=out.index)

    out["_time_precision"] = np.where(
        time_missing | explicit_day,
        "day",
        "time",
    )


    out["_event_datetime"] = pd.NaT
    timed_mask = out["_time_precision"].eq("time")
    out.loc[timed_mask, "_event_datetime"] = pd.to_datetime(
        out.loc[timed_mask, "event_date"].astype(str).str.strip()
        + " "
        + time_text.loc[timed_mask],
        dayfirst=True,
        errors="coerce",
    )
    out.loc[~timed_mask, "_event_datetime"] = out.loc[
        ~timed_mask, "_event_date"
    ]

    out["_window_hours"] = pd.to_numeric(
        out["window_hours"],
        errors="coerce",
    )

    out["_expected_set"] = out[
        "expected_taxonomy"
    ].apply(parse_expected_taxonomy)


    valid_datetime = out["_event_datetime"].notna()
    valid_window = (
        out["_time_precision"].eq("day")
        | out["_window_hours"].gt(0)
    )
    out = out.loc[
        valid_datetime & valid_window
    ].reset_index(drop=True)

    out = out[
        out["_expected_set"].apply(len) > 0
    ].reset_index(drop=True)

    if out.empty:
        raise RuntimeError(
            "No valid events remained after parsing event calendar."
        )

    return out


def normalize_anomaly_flag(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(np.int8)

    numeric = pd.to_numeric(s, errors="coerce")

    if numeric.notna().mean() > 0.95:
        unique = set(numeric.dropna().unique().tolist())

        if unique.issubset({0, 1}):
            return numeric.fillna(0).astype(np.int8)

        return (numeric > 0).astype(np.int8)

    text = s.astype(str).str.strip().str.lower()

    return text.isin(
        {"1", "true", "yes", "anomaly", "anomalous", "outlier"}
    ).astype(np.int8)


def load_anomalies_from_scored_parquets(
    files: list[Path]
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:

    anomaly_parts = []

    data_start = None
    data_end = None

    total_rows = 0
    total_anomalies = 0

    print("\nLoading scored parquet chunks...")

    for i, path in enumerate(files, start=1):

        print(
            f"[{i}/{len(files)}] {path.name}"
        )

        try:
            df = pd.read_parquet(
                path,
                columns=[
                    DATE_COL,
                    ANOMALY_COL,
                    TAXONOMY_COL,
                    COMPONENT_COL,
                ],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read required columns from {path.name}:\n{exc}"
            )

        df[DATE_COL] = pd.to_datetime(
            df[DATE_COL],
            errors="coerce",
        )

        df = df.dropna(subset=[DATE_COL])

        if df.empty:
            continue

        chunk_min = pd.Timestamp(df[DATE_COL].min())
        chunk_max = pd.Timestamp(df[DATE_COL].max())

        data_start = (
            chunk_min
            if data_start is None
            else min(data_start, chunk_min)
        )

        data_end = (
            chunk_max
            if data_end is None
            else max(data_end, chunk_max)
        )

        df["_is_anomaly"] = normalize_anomaly_flag(
            df[ANOMALY_COL]
        )

        total_rows += len(df)

        a = df[
            df["_is_anomaly"] == 1
        ][
            [DATE_COL, TAXONOMY_COL, COMPONENT_COL]
        ].copy()

        a[TAXONOMY_COL] = (
            a[TAXONOMY_COL]
            .astype(str)
            .str.strip()
        )

        a = a[
            a[TAXONOMY_COL].isin(
                VALID_TAXONOMIES
            )
        ]


        if a.empty:
            print(
                f"  rows={len(df):,} "
                "usable_anomalies=0"
            )
            del df, a
            continue

        a[COMPONENT_COL] = a[COMPONENT_COL].fillna("").astype(str)

        def component_mask(row):
            components = {
                x.strip() for x in row[COMPONENT_COL].split(",") if x.strip()
            }


            if not components and row[TAXONOMY_COL] in EXPECTED_COMPONENTS:
                components = {row[TAXONOMY_COL]}
            return sum(
                1 << i for i, tid in enumerate(EXPECTED_COMPONENTS)
                if tid in components
            )

        a["_component_mask"] = a.apply(component_mask, axis=1).astype(np.int8)

        total_anomalies += len(a)

        if not a.empty:
            anomaly_parts.append(a)

        print(
            f"  rows={len(df):,} "
            f"usable_anomalies={len(a):,}"
        )

        del df, a

    if not anomaly_parts:
        raise RuntimeError(
            "No anomalous transactions with T1-T6 taxonomy labels were found."
        )

    anomalies = pd.concat(
        anomaly_parts,
        ignore_index=True,
    )

    anomalies = (
        anomalies
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    print("\nLoaded anomaly taxonomy data:")
    print(f"  total scored rows : {total_rows:,}")
    print(f"  anomaly rows      : {len(anomalies):,}")
    print(f"  data start        : {data_start}")
    print(f"  data end          : {data_end}")

    print("\nTaxonomy counts:")
    print(
        anomalies[TAXONOMY_COL]
        .value_counts()
        .reindex(VALID_TAXONOMIES, fill_value=0)
        .to_string()
    )

    return anomalies, data_start, data_end


def build_taxonomy_prefix_arrays(anomalies: pd.DataFrame):
    times = anomalies[DATE_COL].to_numpy(dtype="datetime64[ns]")
    prefixes = {}
    for tax in VALID_TAXONOMIES:
        values = anomalies[TAXONOMY_COL].eq(tax).to_numpy(dtype=np.int64)
        prefix = np.zeros(len(values) + 1, dtype=np.int64)
        prefix[1:] = np.cumsum(values)
        prefixes[tax] = prefix


    component_masks = anomalies["_component_mask"].to_numpy(dtype=np.int8)
    component_prefixes = {}
    for i, tid in enumerate(EXPECTED_COMPONENTS):
        values = ((component_masks & (1 << i)) != 0).astype(np.int64)
        prefix = np.zeros(len(values) + 1, dtype=np.int64)
        prefix[1:] = np.cumsum(values)
        component_prefixes[tid] = prefix
    match_prefixes = {}
    for expected_mask in range(1, 16):
        values = ((component_masks & expected_mask) != 0).astype(np.int64)
        prefix = np.zeros(len(values) + 1, dtype=np.int64)
        prefix[1:] = np.cumsum(values)
        match_prefixes[expected_mask] = prefix
    total_prefix = np.arange(len(anomalies) + 1, dtype=np.int64)
    return times, total_prefix, prefixes, component_prefixes, match_prefixes


def interval_indices(
    times,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> tuple[int, int]:

    start64 = np.datetime64(
        start.to_datetime64()
    )

    end64 = np.datetime64(
        end.to_datetime64()
    )

    left = np.searchsorted(
        times,
        start64,
        side="left",
    )

    right = np.searchsorted(
        times,
        end64,


        side="left",
    )

    return int(left), int(right)


def event_interval(
    event_datetime: pd.Timestamp,
    window_hours: float,
    time_precision: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if time_precision == "day":
        start = pd.Timestamp(event_datetime).normalize()
        end = start + pd.Timedelta(days=1)
        return start, end

    delta = pd.Timedelta(hours=float(window_hours))
    event_datetime = pd.Timestamp(event_datetime)
    return event_datetime - delta, event_datetime + delta


def interval_match_metrics(
    times,
    taxonomy_prefixes,
    component_prefixes,
    component_match_prefixes,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_set: tuple[str, ...],
) -> dict:

    if end <= start:
        raise ValueError(
            f"Invalid interval: start={start}, end={end}"
        )

    left, right = interval_indices(
        times,
        start,
        end,
    )

    anomaly_tx = int(right - left)

    counts = {}

    for tax in VALID_TAXONOMIES:
        p = taxonomy_prefixes[tax]
        counts[tax] = int(
            p[right] - p[left]
        )

    expected_mask = sum(
        1 << EXPECTED_COMPONENTS.index(t)
        for t in expected_set if t in EXPECTED_COMPONENTS
    )
    if expected_mask == 0:
        raise ValueError("Each event must define at least one expected T1--T4 component")
    match_prefix = component_match_prefixes[expected_mask]
    matched = int(match_prefix[right] - match_prefix[left])

    match_rate = (
        100.0 * matched / anomaly_tx
        if anomaly_tx > 0
        else np.nan
    )

    dominant_taxonomy = (
        max(
            counts,
            key=counts.get,
        )
        if anomaly_tx > 0
        else None
    )

    component_counts = {
        tid: int(prefix[right] - prefix[left])
        for tid, prefix in component_prefixes.items()
    }
    dominant_component = (
        max(component_counts, key=component_counts.get)
        if anomaly_tx > 0
        else None
    )

    return {
        "window_start": start,
        "window_end": end,
        "window_duration_hours": (
            end - start
        ).total_seconds() / 3600.0,
        "anomaly_tx": anomaly_tx,
        "matched_anomaly_tx": matched,
        "taxonomy_match_rate": match_rate,
        "dominant_taxonomy": dominant_taxonomy,
        "dominant_component": dominant_component,
        "match_definition": "any expected atomic T1-T4 component",
        **{
            f"{tax}_count": counts[tax]
            for tax in VALID_TAXONOMIES
        },
        **{
            f"{tax}_component_count": component_counts[tax]
            for tax in EXPECTED_COMPONENTS
        },
    }


def split_events_by_observation_period(
    events: pd.DataFrame,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    valid_rows = []
    excluded_rows = []

    for _, row in events.iterrows():

        event_time = pd.Timestamp(
            row["_event_datetime"]
        )

        start, end = event_interval(
            event_datetime=event_time,
            window_hours=float(row["_window_hours"]),
            time_precision=str(row["_time_precision"]),
        )

        row2 = row.copy()

        row2["_window_start"] = start
        row2["_window_end"] = end

        if (
            start >= data_start
            and end <= data_end
        ):
            valid_rows.append(row2)
        else:
            row2["_exclusion_reason"] = (
                "complete event window falls outside "
                "the observed transaction period"
            )
            excluded_rows.append(row2)

    valid = pd.DataFrame(valid_rows)
    excluded = pd.DataFrame(excluded_rows)

    return (
        valid.reset_index(drop=True),
        excluded.reset_index(drop=True),
    )


def benjamini_hochberg(
    p_values: np.ndarray
) -> np.ndarray:
    p = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = (
        ranked
        * n
        / np.arange(1, n + 1)
    )


    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    out = np.empty(
        n,
        dtype=float,
    )

    out[order] = adjusted

    return out


def random_interval_start(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    duration: pd.Timedelta,
    rng: np.random.Generator,
) -> pd.Timestamp:

    earliest = data_start
    latest = data_end - duration

    if latest < earliest:
        raise RuntimeError(
            f"Window duration {duration} does not fit "
            "inside the observed period."
        )

    span_seconds = int(
        (latest - earliest)
        .total_seconds()
    )

    offset = int(
        rng.integers(
            0,
            span_seconds + 1,
        )
    )

    return earliest + pd.Timedelta(seconds=offset)


def run_event_level_null_model(
    anomalies: pd.DataFrame,
    valid_events: pd.DataFrame,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    n_permutations: int,
    random_state: int,
):
    (
        times,
        _total_prefix,
        taxonomy_prefixes,
        component_prefixes,
        component_match_prefixes,
    ) = build_taxonomy_prefix_arrays(
        anomalies
    )

    rng = np.random.default_rng(
        random_state
    )

    observed_rows = []
    null_rows = []
    summary_rows = []

    for event_idx, event in valid_events.iterrows():

        event_id = str(
            event["event_id"]
        )

        event_name = str(
            event.get(
                "event_name",
                ""
            )
        )

        event_time = pd.Timestamp(
            event["_event_datetime"]
        )

        window_hours = float(
            event["_window_hours"]
        )

        time_precision = str(
            event["_time_precision"]
        )

        observed_start, observed_end = event_interval(
            event_datetime=event_time,
            window_hours=window_hours,
            time_precision=time_precision,
        )

        interval_duration = (
            observed_end - observed_start
        )

        interval_duration_hours = (
            interval_duration.total_seconds() / 3600.0
        )

        expected_set = tuple(
            event["_expected_set"]
        )

        expected_string = ",".join(
            expected_set
        )


        obs = interval_match_metrics(
            times=times,
            taxonomy_prefixes=taxonomy_prefixes,
            component_prefixes=component_prefixes,
            component_match_prefixes=component_match_prefixes,
            start=observed_start,
            end=observed_end,
            expected_set=expected_set,
        )

        observed_rate = float(
            obs["taxonomy_match_rate"]
        )

        observed_rows.append({
            "event_id": event_id,
            "event_name": event_name,
            "event_datetime": event_time,
            "time_precision": time_precision,
            "window_hours": window_hours,
            "effective_window_hours": interval_duration_hours,
            "expected_taxonomy": expected_string,
            **obs,
        })

        print(
            f"\n{event_id}: "
            f"observed Match={observed_rate:.3f}% "
            f"anomaly_tx={obs['anomaly_tx']:,} "
            f"expected={expected_string} "
            f"precision={time_precision} "
            f"duration={interval_duration_hours:g}h"
        )


        null_rates = np.empty(
            n_permutations,
            dtype=float,
        )

        null_anomaly_counts = np.empty(
            n_permutations,
            dtype=np.int64,
        )

        for b in range(
            n_permutations
        ):

            random_start = random_interval_start(
                data_start=data_start,
                data_end=data_end,
                duration=interval_duration,
                rng=rng,
            )

            random_end = random_start + interval_duration

            m = interval_match_metrics(
                times=times,
                taxonomy_prefixes=taxonomy_prefixes,
                component_prefixes=component_prefixes,
                component_match_prefixes=component_match_prefixes,
                start=random_start,
                end=random_end,
                expected_set=expected_set,
            )

            null_rates[b] = m[
                "taxonomy_match_rate"
            ]

            null_anomaly_counts[b] = m[
                "anomaly_tx"
            ]

            null_rows.append({
                "event_id": event_id,
                "permutation": b + 1,
                "random_window_start": random_start,
                "random_window_end": random_end,
                "time_precision": time_precision,
                "window_hours": window_hours,
                "effective_window_hours": interval_duration_hours,
                "expected_taxonomy": expected_string,
                "null_anomaly_tx": m["anomaly_tx"],
                "null_matched_anomaly_tx":
                    m["matched_anomaly_tx"],
                "null_taxonomy_match_rate":
                    m["taxonomy_match_rate"],
            })


        finite = np.isfinite(
            null_rates
        )

        valid_null = null_rates[
            finite
        ]

        if len(valid_null) == 0:
            raise RuntimeError(
                f"No valid null windows with anomalies "
                f"were generated for {event_id}."
            )


        empirical_p = (
            1
            + np.sum(
                valid_null
                >= observed_rate
            )
        ) / (
            len(valid_null)
            + 1
        )

        null_mean = float(
            np.mean(valid_null)
        )

        null_median = float(
            np.median(valid_null)
        )

        ci_low = float(
            np.percentile(
                valid_null,
                2.5
            )
        )

        ci_high = float(
            np.percentile(
                valid_null,
                97.5
            )
        )

        percentile = float(
            np.mean(
                valid_null
                < observed_rate
            )
            * 100.0
        )

        effect_vs_null_mean = (
            observed_rate
            - null_mean
        )

        summary_rows.append({
            "event_id": event_id,
            "event_name": event_name,
            "event_datetime": event_time,
            "time_precision": time_precision,
            "window_hours": window_hours,
            "effective_window_hours": interval_duration_hours,
            "observed_window_start": observed_start,
            "observed_window_end": observed_end,
            "expected_taxonomy": expected_string,
            "observed_anomaly_tx":
                int(obs["anomaly_tx"]),
            "observed_matched_anomaly_tx":
                int(obs["matched_anomaly_tx"]),
            "observed_match_rate_percent":
                observed_rate,
            "null_mean_match_rate_percent":
                null_mean,
            "null_median_match_rate_percent":
                null_median,
            "null_ci_2_5_percent":
                ci_low,
            "null_ci_97_5_percent":
                ci_high,
            "effect_vs_null_mean_pp":
                effect_vs_null_mean,
            "observed_percentile_in_null":
                percentile,
            "empirical_p_value_one_sided":
                float(empirical_p),
            "valid_null_permutations":
                int(len(valid_null)),
            "zero_anomaly_null_windows":
                int(
                    np.sum(
                        ~finite
                    )
                ),
        })

        print(
            f"  null mean={null_mean:.3f}% "
            f"central 95% null interval="
            f"[{ci_low:.3f}, {ci_high:.3f}] "
            f"p={empirical_p:.6f}"
        )

    observed_df = pd.DataFrame(
        observed_rows
    )

    null_df = pd.DataFrame(
        null_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df[
        "fdr_bh_q_value"
    ] = benjamini_hochberg(
        summary_df[
            "empirical_p_value_one_sided"
        ].to_numpy()
    )

    summary_df[
        "significant_p_lt_0_05"
    ] = (
        summary_df[
            "empirical_p_value_one_sided"
        ]
        < 0.05
    )

    summary_df[
        "significant_fdr_q_lt_0_05"
    ] = (
        summary_df[
            "fdr_bh_q_value"
        ]
        < 0.05
    )

    return (
        observed_df,
        null_df,
        summary_df,
    )


def save_summary_plot(
    summary: pd.DataFrame,
    output_path: Path
):
    try:
        import matplotlib.pyplot as plt

        s = summary.copy()

        x = np.arange(
            len(s)
        )

        observed = s[
            "observed_match_rate_percent"
        ].to_numpy()

        null_median = s[
            "null_median_match_rate_percent"
        ].to_numpy()

        low = s[
            "null_ci_2_5_percent"
        ].to_numpy()

        high = s[
            "null_ci_97_5_percent"
        ].to_numpy()

        yerr = np.vstack([
            null_median - low,
            high - null_median,
        ])

        plt.figure(
            figsize=(10, 5.5)
        )

        plt.errorbar(
            x,
            null_median,
            yerr=yerr,
            fmt="o",
            capsize=4,
            label="Null median and 2.5th-97.5th percentile range",
        )

        plt.scatter(
            x,
            observed,
            marker="x",
            s=70,
            label="Observed match rate",
        )

        plt.xticks(
            x,
            s["event_id"].astype(str)
        )

        plt.ylabel(
            "Taxonomy match rate (%)"
        )

        plt.xlabel(
            "Event"
        )

        plt.title(
            "Observed Event Taxonomy Match vs Random-Time Null"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    except Exception as exc:
        print(
            f"\nWARNING: plot generation failed: {exc}"
        )


def write_method_notes(
    path: Path,
    parquet_files: list[Path],
    summary: pd.DataFrame,
    excluded: pd.DataFrame,
    n_permutations: int,
    random_state: int,
):
    excluded_ids = (
        ", ".join(
            excluded[
                "event_id"
            ]
            .astype(str)
            .tolist()
        )
        if not excluded.empty
        else "none"
    )

    text = f"""EVENT-LEVEL TAXONOMY-MATCH NULL MODEL
=====================================

Source parquet pattern:
{DEFAULT_PARQUET_PATTERN}

Number of scored parquet chunks:
{len(parquet_files)}

Required parquet columns:
date
final_anomaly
taxonomy_id

Taxonomy match metric
---------------------
For event e:

    MatchRate_e =
        N_matched_anomalies,e
        --------------------- x 100
        N_anomalies,e

An anomalous transaction is counted as matched when its taxonomy_id belongs
to the predefined expected_taxonomy set for that event.

If an event has multiple expected taxonomy labels, matching ANY one of those
labels is sufficient for that transaction to be counted as matched.

Null model
----------
Events with a known time use a symmetric interval centered on event_time:
event_time +/- window_hours. Events marked time_precision=day, or events with
an empty event_time, use the complete calendar day [00:00, next-day 00:00).
For day-level events, window_hours is ignored and the effective duration is
exactly 24 hours.

For each eligible event separately, random interval starts are drawn uniformly
from the observed transaction period, subject to the complete interval fitting
inside that period. Every random interval has exactly the same duration as the
corresponding observed interval. Thus, a day-level event is compared with
randomly positioned 24-hour intervals.

For every random window, the same taxonomy match rate is recomputed using the
same event-specific expected taxonomy set.

Number of permutations per event:
{n_permutations}

Random state:
{random_state}

Event-level empirical one-sided p-value:
    p_e = (1 + # null MatchRate_e >= observed MatchRate_e) / (B + 1)

The one-sided test asks whether the observed taxonomy match rate is unusually
HIGH relative to randomly positioned windows.

Multiple testing
----------------
Because one hypothesis test is performed for each eligible event,
Benjamini-Hochberg FDR-adjusted q-values are reported in addition to the
raw empirical p-values.

Events excluded because their complete window is outside the observation period:
{excluded_ids}

Interpretation
--------------
A small event-specific p-value indicates that the event's observed taxonomy
match rate is higher than expected from randomly positioned windows using the
same window duration and expected taxonomy set.

The reported 2.5th and 97.5th percentiles form the central 95% range of the
simulated null distribution; they are not a confidence interval for an
estimated population parameter.

This analysis tests event-taxonomy correspondence relative to a random-time
baseline. It does NOT establish causality.
"""

    path.write_text(
        text,
        encoding="utf-8",
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Event-level taxonomy-match permutation null model "
            "for the Dogecoin anomaly taxonomy."
        )
    )

    p.add_argument(
        "--workdir",
        default=DEFAULT_WORKDIR,
    )

    p.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
    )

    p.add_argument(
        "--events-file",
        default=None,
    )

    p.add_argument(
        "--parquet-pattern",
        default=DEFAULT_PARQUET_PATTERN,
    )

    p.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
    )

    p.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    return p.parse_args()


def main():
    args = parse_args()

    workdir = Path(
        args.workdir
    ).expanduser().resolve()

    if not workdir.exists():
        print(
            f"ERROR: workdir does not exist:\n{workdir}"
        )
        sys.exit(1)

    output_dir = (
        workdir
        / args.output_subdir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "EVENT-LEVEL TAXONOMY-MATCH "
        "PERMUTATION NULL MODEL"
    )
    print("=" * 80)

    print(
        f"Workdir         : {workdir}"
    )

    print(
        f"Output dir      : {output_dir}"
    )

    print(
        f"Parquet pattern : {args.parquet_pattern}"
    )

    print(
        f"Permutations    : {args.n_permutations} per event"
    )

    print(
        f"Random state    : {args.random_state}"
    )


    event_file = detect_event_file(
        workdir,
        args.events_file,
    )

    events = load_event_calendar(
        event_file
    )

    parquet_files = detect_scored_parquets(
        workdir,
        args.parquet_pattern,
    )

    print(
        f"\nEvent file       : {event_file.name}"
    )

    print(
        f"Events parsed    : {len(events)}"
    )

    print(
        f"Scored parquets  : {len(parquet_files)}"
    )


    (
        anomalies,
        data_start,
        data_end,
    ) = load_anomalies_from_scored_parquets(
        parquet_files
    )


    (
        valid_events,
        excluded_events,
    ) = split_events_by_observation_period(
        events,
        data_start,
        data_end,
    )

    print(
        f"\nEligible events  : {len(valid_events)}"
    )

    print(
        f"Excluded events  : {len(excluded_events)}"
    )

    if not excluded_events.empty:
        print(
            "Excluded IDs     : "
            + ", ".join(
                excluded_events[
                    "event_id"
                ]
                .astype(str)
                .tolist()
            )
        )

    if valid_events.empty:
        raise RuntimeError(
            "No eligible event windows remained."
        )


    (
        observed_df,
        null_df,
        summary_df,
    ) = run_event_level_null_model(
        anomalies=anomalies,
        valid_events=valid_events,
        data_start=data_start,
        data_end=data_end,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )


    summary_path = (
        output_dir
        / "taxonomy_match_null_summary.csv"
    )

    null_path = (
        output_dir
        / "taxonomy_match_null_distribution.csv"
    )

    observed_path = (
        output_dir
        / "taxonomy_match_observed_events.csv"
    )

    excluded_path = (
        output_dir
        / "taxonomy_match_excluded_events.csv"
    )

    notes_path = (
        output_dir
        / "taxonomy_match_method_notes.txt"
    )

    plot_path = (
        output_dir
        / "taxonomy_match_null_plot.png"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    null_df.to_csv(
        null_path,
        index=False,
    )

    observed_df.to_csv(
        observed_path,
        index=False,
    )

    excluded_events.to_csv(
        excluded_path,
        index=False,
    )

    write_method_notes(
        notes_path,
        parquet_files,
        summary_df,
        excluded_events,
        args.n_permutations,
        args.random_state,
    )

    save_summary_plot(
        summary_df,
        plot_path,
    )


    print("\n" + "=" * 80)
    print("EVENT-LEVEL TAXONOMY-MATCH NULL RESULTS")
    print("=" * 80)

    display_cols = [
        "event_id",
        "expected_taxonomy",
        "observed_match_rate_percent",
        "null_mean_match_rate_percent",
        "effect_vs_null_mean_pp",
        "empirical_p_value_one_sided",
        "fdr_bh_q_value",
    ]

    print(
        summary_df[
            display_cols
        ]
        .round(4)
        .to_string(index=False)
    )

    n_raw_sig = int(
        summary_df[
            "significant_p_lt_0_05"
        ].sum()
    )

    n_fdr_sig = int(
        summary_df[
            "significant_fdr_q_lt_0_05"
        ].sum()
    )

    print("\n" + "-" * 80)

    print(
        f"Raw p < 0.05 events : "
        f"{n_raw_sig}/{len(summary_df)}"
    )

    print(
        f"FDR q < 0.05 events : "
        f"{n_fdr_sig}/{len(summary_df)}"
    )

    print("-" * 80)

    print(
        "\nInterpretation guide:"
    )

    print(
        "  p < 0.05: observed event-taxonomy match is higher "
        "than expected under random-time windows."
    )

    print(
        "  q < 0.05: the result remains significant after "
        "Benjamini-Hochberg correction across events."
    )

    print(
        "  Non-significant results should be reported as "
        "contextual/semantic correspondence rather than "
        "evidence of above-chance event-specific alignment."
    )

    print(
        "\nAll outputs saved under:"
    )

    print(
        output_dir
    )

    print("\nGenerated files:")

    for p in sorted(
        output_dir.iterdir()
    ):
        if p.is_file():
            print(
                f"  - {p.name}"
            )


if __name__ == "__main__":
    main()