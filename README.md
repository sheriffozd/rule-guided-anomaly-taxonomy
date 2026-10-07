# Rule-Guided Anomaly Taxonomy for Large-Scale Dogecoin Transaction Graphs

Research code and result tables accompanying the manuscript **Rule-Guided Anomaly Taxonomy for Large-Scale Dogecoin Transaction Graphs**.

The workflow constructs a Dogecoin transaction graph through Dogecoin Core, Kafka, and Neo4j; extracts temporal, behavioral, and structural features; detects anomalies with Isolation Forest; and characterizes them using explicit behavioral profiles. Additional scripts compare five detectors, evaluate event-window correspondence against a random-time null model, compare human and LLM semantic expectations, and measure profile lift and address concentration.

## Repository layout

| Directory | Contents |
|---|---|
| `datatransmission/` | Dogecoin Core producer and Neo4j ingestion consumers |
| `anomalydetection/` | Detection, taxonomy, benchmark, and evaluation scripts |
| `resources/` | Published result CSVs and supporting analysis inputs |
| `resources/work_tx_address_gds13_train_centrality/` | Runtime working directory for feature chunks, scored chunks, profile thresholds, event calendar, and generated outputs |

The uploaded filenames below include `(1)` suffixes. Commands use those names exactly. If files are renamed, update the commands accordingly.

## Data and observation unit

- Study period: May 1–22, 2021, in UTC.
- Training: records before May 17, 2021.
- Validation: May 17, 2021.
- Test: May 18–22, 2021.
- Reported dataset: 16,905,794 derived input–output association records.

Each resolved input address is paired with every eligible output address of its source blockchain transaction. Consequently, one blockchain transaction may generate multiple records. Each association retains the output value, so summing association values does not measure net transferred volume. These records are neither distinct payments nor unique blockchain transactions.

The `date` field is the containing block's timestamp, converted to UTC. It is not the time of transaction broadcast. Records from the same block share this timestamp. Temporal activity features use completed, fixed UTC-aligned intervals.

Graph centralities use training-period relationships and remain fixed for validation and test scoring. The projection can include address nodes stored outside the training period as isolated nodes. Scaling, detector fitting/calibration, and profile threshold estimation use training-period observations.

## Behavioral profiles

Thresholds are saved in `profile_thresholds.json`. The baseline uses training-reference 95th percentiles.

| Profile | Condition |
|---|---|
| T1: High Transaction Density | Positive 15-minute transaction-count deviation score meeting its training q95 threshold |
| T2: High-Value Transfer | Positive stored log-transformed output value meeting its training q95 threshold |
| T3: High Sender Participation | Positive log-transformed active-sender count meeting its training q95 threshold |
| T4: High-Centrality Address Involvement | At least two of degree, PageRank, and eigenvector centrality meet their metric-specific training q95 thresholds for the same sender or receiver endpoint |

Centrality thresholds use pooled sender and receiver reference values for each metric. A record satisfying exactly one profile receives that profile's label. Two or more matches produce **T5 (Multi-Profile Overlap)**; no matches produce **T6 (Uncharacterized Anomaly)**. Atomic T1–T4 matches remain available for overlap-aware analyses. Some scripts use “Composite Profile” for T5; this denotes the same multiple-profile assignment.

The final taxonomy is rule-guided and does not use HDBSCAN or UMAP. Profiles characterize detector-selected anomalies; they do not establish fraud, address ownership, or causal attribution of detector scores.

## Software and services

The workflow requires Python, Dogecoin Core with historical RPC access, Kafka, and Neo4j with Graph Data Science. The node-ingestion consumer also requires APOC. The LLM evaluation uses a local Ollama service and `qwen2.5:7b` through its OpenAI-compatible endpoint.

Python packages imported by the scripts include:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy scikit-learn pyod duckdb joblib pyarrow matplotlib neo4j kafka-python requests openai yfinance
```

This is a dependency list, not a verified version lock. Record the exact Python, package, Dogecoin Core, Kafka, Neo4j, GDS, APOC, and Ollama versions used in a replication. `pyarrow` provides Parquet support. The market-data functions also import `yfinance`; verify that the chosen input remains the CoinGecko DOGE/USD series used in the manuscript.

## Configure before running

Run analysis commands from the repository root so relative paths resolve consistently.

1. Set service addresses and credentials in the relevant scripts. Remove real passwords and tokens before publishing the source.
2. Point all analysis scripts to `resources/work_tx_address_gds13_train_centrality/`. Scripts 1–5 already reference this location. Script 6 and the consistency audit currently reference `./work_tx_address_gds13_train_centrality`; script 7 contains an absolute workstation path. Update these constants, including their derived output paths, before running.
3. Place the final `event_calendar.csv` in that working directory. It must preserve the event descriptions, expected T1–T4 sets, UTC dates/times, window parameters, and `time_precision` used in the manuscript.
4. Keep each family of Parquet chunks from one complete run. Do not mix older taxonomy assignments or overlapping exports in the same working directory.
5. Confirm settings in the supplied source: the primary IF uses 150 estimators, a 500,000-record fitting sample, a 500,000-record calibration sample, contamination 0.015, and random state 42. The separate benchmark uses its own fitting and calibration protocol; its IF result is not the primary IF run.

### Ingestion integration checks

The supplied ingestion scripts require configuration and integration before they form a runnable end-to-end pipeline:

- The producer uses topic `doge-tx`; both consumers use `dogecoin-transactions`. Set the same topic and Kafka brokers in all three files.
- The producer emits `txid`, `from`, `to`, `value`, `date`, and `block`. Consumers expect an additional association-level `id`. Implement a stable record identifier and preserve it across both consumer passes. A transaction hash alone is not unique for the expanded input–output records; repeated input entries must remain distinguishable.
- The extensionless file `kafka_to_neo4j_relationships_threaded` actually creates nodes. The `.py` file with the similar name creates relationships. They are different stages, not duplicate files.
- The relationship consumer uses `MATCH` for previously created nodes. Complete node ingestion before replaying records through the relationship consumer; otherwise missing matches can omit relationships.
- Current consumer stopping limits are 2,000 messages for node ingestion and 454 for relationship ingestion. Adjust them for a complete dataset and verify consumed counts and Kafka offsets.
- The producer's block range is hard-coded. Select the heights covering the entire intended UTC study period and verify the resulting timestamps.

These checks describe the supplied source; publishing result CSVs does not resolve ingestion configuration automatically.

## Execution order

### 1. Build the graph

Configure the checks above, then run the producer and the node consumer. The node consumer needs the resulting Kafka messages; start it while production proceeds or consume the retained topic afterward.

```bash
python 'datatransmission/dogecoin_core_to_kafka_parallel.py'
python 'datatransmission/kafka_to_neo4j_relationships_threaded'
```

After node ingestion completes, run the relationship pass with its own consumer group and verified starting offsets:

```bash
python 'datatransmission/kafka_to_neo4j_relationships_threaded.py'
```

Graph labels used by analysis include `RawTransaction`, `Address`, `Block`, `Date`, and `TxID`; relationships include `SENT`, `RECEIVED`, `IN_BLOCK`, `ON_DATE`, and `HAS_ID`.

### 2. Primary anomaly detection and taxonomy

```bash
python 'anomalydetection/1_dogecoin_if_taxonomy_pipeline.py'
```

This entry point recomputes training centralities, exports and sorts records, builds features, fits/calibrates IF, estimates profile thresholds, assigns T1–T6, and generates summary, event, and market outputs. It is a full rebuild, not a CSV-only report command.

Important generated inputs for later stages include `fe_chunk_*.parquet`, `base_scored_chunk_*.parquet`, `scored_chunk_*.parquet`, and `profile_thresholds.json`.

### 3. Detector benchmark and taxonomy sensitivity

```bash
python 'anomalydetection/2_dogecoin_detector_benchmark.py'
python 'anomalydetection/3_dogecoin_taxonomy_detector_sensitivity.py'
```

The benchmark compares IF, LOF, OCSVM, COPOD, and ECOD on the shared feature representation and creates `baseline_scored_chunk_*.parquet` and pairwise overlap reports. Script 3 applies shared taxonomy thresholds to those detector outputs and saves distributions in `detector_taxonomy_robustness/` under the working directory.

### 4. Event-specific random-time null analysis

```bash
python 'anomalydetection/4_dogecoin_event_taxonomy_null_analysis.py' \
  --workdir resources/work_tx_address_gds13_train_centrality \
  --events-file resources/work_tx_address_gds13_train_centrality/event_calendar.csv \
  --n-permutations 10000 \
  --random-state 42
```

The default scored input pattern is `scored_chunk_*.parquet`. Outputs are saved in `taxonomy_match_null_results/`.

Match rate counts anomalies satisfying at least one expected atomic T1–T4 profile, including profiles retained within T5; each anomaly counts once. Time-specific windows are symmetric (±h, total 2h). Day-level events use the full UTC calendar day, while their null reference intervals are randomly positioned 24-hour windows.

The empirical one-sided p-value is `(1 + k) / (1 + B)`, where `k` counts valid null match rates at least as large as the observed rate. Benjamini–Hochberg correction is applied across events. The null 2.5th–97.5th percentile interval is a distribution summary, not a confidence interval for an observed event effect.

### 5. Auxiliary LLM evaluation

Start Ollama, make `qwen2.5:7b` available, and run:

```bash
python 'anomalydetection/5_dogecoin_llm_semantic_evaluation.py'
```

The script uses `http://localhost:11434/v1` and temperature 0. It compares human and LLM expected T1–T4 sets; these are semantic expectations rather than transaction ground-truth labels. Inspect evaluation failures: the supplied implementation can return a T1 fallback with confidence 0 after retries. Such entries must be identified before interpreting agreement metrics.

### 6. Lift, concentration, and assignment audit

After aligning their working-directory constants:

```bash
python 'anomalydetection/6_dogecoin_profile_lift_analysis.py'
python 'anomalydetection/7_dogecoin_t4_address_concentration.py'
python 'anomalydetection/dogecoin_taxonomy_consistency_audit.py'
```

Lift requires **all** `base_scored_chunk_*.parquet` records, including IF-normal records, plus `profile_thresholds.json`. NORMAL denotes records not selected by IF; it is not an independently verified normal label. Lift is `P(profile | anomaly) / P(profile | NORMAL)`.

Concentration additionally requires access to the original Neo4j graph to resolve endpoints by `graph_id`; arbitrary graph reconstruction or re-export can invalidate this mapping. Its connection settings use `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` or an interactive password prompt.

The audit reads `scored_chunk_*.parquet` and checks final labels and retained component counts.

## Published results in resources/

Use the final CSVs listed below. Parenthesized download suffixes and run timestamps identify the supplied files; renamed copies should retain provenance in a manifest.

| Manuscript result | Supplied CSV |
|---|---|
| Table 7: final taxonomy | `final_taxonomy_summary(20261007-232652).csv` |
| Table 8: atomic component prevalence | `taxonomy_component_combinations(3).csv` and `profile_lift_overall(1).csv` |
| Daily taxonomy | `daily_taxonomy_counts(8).csv`, `daily_taxonomy_percentages(8).csv` |
| Table 9: profile lift | `profile_lift_by_split(1).csv` |
| Table 10: T4 concentration | `t4_address_concentration_summary(1).csv`, `t4_address_rankings(1).csv` |
| Table 11: event-window results | `event_grounded_validation_results(20261007-232650).csv`, `taxonomy_match_observed_events(20261007-232939).csv` |
| Event null analysis | `taxonomy_match_null_summary(20261007-232938).csv`, `taxonomy_match_null_distribution(20261007-232948).csv` |
| Excluded event | `taxonomy_match_excluded_events(20261007-232655).csv` |
| Table 12: detector distributions | `detector_taxonomy_summary(10).csv` |
| Table 13: detector agreement | `baseline_pairwise_jaccard(4).csv` (`ALL` rows) |
| Table 14: market association | `taxonomy_market_daily_merged(20261007-232655).csv`, `taxonomy_market_lag_correlation(20261007-232655).csv` |
| Table 15: human–LLM agreement | `llm_auxiliary_semantic_validation(8).csv` |
| Taxonomy consistency | `taxonomy_overlap_audit(2).csv` |

The `232657` null summary/distribution and `232658` observed-events files are superseded outputs from an older setup. Exclude them from the published final result set.

### Event identifier mapping

The supplied event CSVs and manuscript use different identifiers for four events. Until identifiers are harmonized across every event-related file, use this mapping:

| Event | CSV identifier | Manuscript identifier |
|---|---|---|
| Tesla had not sold Bitcoin, May 17 | E11 | E10 |
| Chinese financial-industry restrictions, May 18 | E12 | E11 |
| Broad cryptocurrency sell-off, May 19 | E13 | E12 |
| “I haven't & won't sell any Doge,” May 20 | E10 | E13 |

Other identifiers are unchanged. Do not join files by event ID across these two conventions without applying the mapping.

The supplied event-grounded results file contains a `NORMAL_count` column populated with zeros. Do not interpret it as the number of normal records; derive that number as `total_tx - anomaly_tx`, or omit the column in the published copy.

## Reference checks

| Check | Expected result |
|---|---|
| Primary IF anomalies | 238,824 (approximately 1.413% of records) |
| Final T1 / T2 / T3 | 10,062 / 1,256 / 245 |
| Final T4 / T5 / T6 | 125,642 / 99,748 / 1,871 |
| Atomic T1 / T2 / T3 / T4 | 67,046 / 41,882 / 55,448 / 212,276 |
| Audit label/component-count mismatches | 0 / 0 |
| Benchmark IF anomalies | 219,075, a separate IF fit from the primary run |
| COPOD–ECOD / IF–OCSVM Jaccard | 73.02% / 58.30% |
| Test-period T2 / T4 lift | 8.747 / 65.101 |
| Event null analysis | 13 eligible events; 10,000 reference windows each; minimum p approximately 0.0823, minimum BH q approximately 0.7064 |
| Human–LLM agreement | 14 events; mean Jaccard 0.4524; micro-F1 0.5490; 5 exact, 4 partial, 5 none |

No event was significant under the specified random-time test. Market correlations are exploratory, and the LLM evaluation is an auxiliary semantic comparison. None of these analyses establishes causality or transaction-level detection accuracy.

## Reproducibility and availability

Result CSVs support inspection of the published findings. They are insufficient to rerun detection or null sampling by themselves. Full replication additionally needs the original graph or reconstructed association records, the feature/scored Parquet chunks, the final event calendar, profile thresholds, and exact software/configuration versions. Identify which of these inputs are included with a repository release and provide acquisition or regeneration instructions for the rest.

The full-scale processing workflow was not executed as part of preparing this README. Runtime depends on graph size, service configuration, available memory, and storage; no replication runtime is asserted.

## Citation and license

Add the public repository URL and any issued software DOI to `CITATION.cff`. Add the article DOI after it is assigned. Choose and include a `LICENSE` file before describing the repository as open source; this README does not assign a license.
