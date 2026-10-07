import os
import requests
import yfinance as yf
import pandas as pd

os.environ["MALLOC_CONF"] = "background_thread:false,narenas:1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

import time
import duckdb
import numpy as np
import pandas as pd

from neo4j import GraphDatabase
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

try:
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD

    HAS_PYOD = True
except Exception:
    HAS_PYOD = False

try:
    import joblib

    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7688")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Password123")

TIMEOUT = 3600
WRITE_BATCH = 5000

INPUT_DIR = "./resources/work_tx_address_gds13_train_centrality"
OUTPUT_DIR = "./resources/work_tx_address_gds13_train_centrality"
FEATURE_INPUT_DIR = "./resources/work_tx_address_gds13_train_centrality"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_FMT = "%Y-%m-%d %H:%M:%S"

IF_FIT_SAMPLE = 500_000
IF_N_ESTIMATORS = 150
IF_CONTAMINATION = 0.015
IF_RANDOM_STATE = 42

# All benchmark detectors share the same training sample.

BASELINE_FIT_SAMPLE = 30_000
BASELINE_CALIBRATION_SAMPLE = 100_000
BASELINE_CONTAMINATION = IF_CONTAMINATION
BASELINE_RANDOM_STATE = 42

LOF_N_NEIGHBORS = 35
OCSVM_NU = BASELINE_CONTAMINATION
OCSVM_KERNEL = "rbf"
OCSVM_GAMMA = "scale"
OCSVM_MAX_TRAIN_ROWS = BASELINE_FIT_SAMPLE

BENCHMARK_DETECTORS = ["IF", "LOF", "OCSVM", "COPOD", "ECOD"]

WRITE_ONLY_ANOMALIES = False

IF_LABEL_COLUMN = "anomaly_label_gds"
IF_SCORE_COLUMN = "anomaly_score_gds"

RUN_GDS = False
RUN_EIGENVECTOR = True

CENTRALITY_COLUMNS = [
    "sender_degree",
    "sender_pageRank",
    "sender_eigenvector",
    "receiver_degree",
    "receiver_pageRank",
    "receiver_eigenvector",
]

MODEL_FEATURE_COLUMNS = [
    "value",
    "seconds_since_last_log",
    "tx_count_15min_log",
    "tx_count_60min_log",
    "tx_count_15min_delta_abs_log",
    "tx_count_15min_z",
    "value_std_15m_log",
    "active_sender_15m_log",
    "max_degree_log",
    "max_pageRank_log",
    "max_eigenvector_log",
    "sender_degree_log",
    "receiver_degree_log",
    "sender_pageRank_log",
    "receiver_pageRank_log",
    "sender_eigenvector_log",
    "receiver_eigenvector_log",
    "same_timestamp_ratio_15m",
    "same_timestamp_count",
]

TAXONOMY_MAP = {
    "T1": "High Transaction Density",
    "T2": "High-Value Transfer",
    "T3": "High Sender Participation",
    "T4": "High-Centrality Address Involvement",
    "T5": "Composite Profile",
    "T6": "Uncharacterized Anomaly",
    "NORMAL": "Normal",
}

TRAIN_END = "2021-05-17"
VALID_END = "2021-05-18"
TEST_START = "2021-05-18"


def ensure_indexes(driver):
    queries = [
        "CREATE INDEX raw_tx_graph_id IF NOT EXISTS FOR (t:RawTransaction) ON (t.graph_id)",
        "CREATE INDEX raw_tx_date IF NOT EXISTS FOR (t:RawTransaction) ON (t.date)",
        "CREATE INDEX raw_tx_id IF NOT EXISTS FOR (t:RawTransaction) ON (t.id)",
        "CREATE INDEX tx_id IF NOT EXISTS FOR (t:Transaction) ON (t.id)",
        "CREATE INDEX address_addr_graph_id IF NOT EXISTS FOR (a:Address) ON (a.addr_graph_id)",
        "CREATE INDEX address_address IF NOT EXISTS FOR (a:Address) ON (a.address)",
    ]

    with driver.session() as session:
        for q in queries:
            try:
                session.run(q, timeout=TIMEOUT).consume()
                print("[OK] index:", q)
            except Exception as e:
                print("[WARN] index error:", e)


def ensure_address_graph_id(driver):
    print("[INFO] Ensuring Address.addr_graph_id...")

    query = """
    CALL apoc.periodic.iterate(
      "
      MATCH (a:Address)
      WHERE a.addr_graph_id IS NULL
      RETURN a
      ",
      "
      SET a.addr_graph_id = id(a)
      ",
      {batchSize: 20000, parallel: false}
    )
    YIELD batches, total
    RETURN batches, total
    """

    with driver.session() as session:
        rec = session.run(query, timeout=TIMEOUT).single()
        print("[OK] Address.addr_graph_id:", dict(rec))


def drop_gds_graph(driver):
    query = """
    CALL gds.graph.drop('address_graph', false)
    YIELD graphName
    RETURN graphName
    """

    with driver.session() as session:
        try:
            session.run(query, timeout=TIMEOUT).consume()
            print("[OK] old address_graph dropped")
        except Exception as e:
            print("[INFO] no graph dropped:", e)


def project_graph(driver):
    print("[INFO] Creating Address -> Address GDS projection...")

    drop_gds_graph(driver)

    query = """
    CALL gds.graph.project.cypher(
      'address_graph',

      '
      MATCH (a:Address)
      RETURN 
        id(a) AS id,
        a.addr_graph_id AS addr_graph_id
      ',

      '
      MATCH (a:Address)-[:SENT]->(:RawTransaction)-[:RECEIVED]->(b:Address)
      RETURN 
        id(a) AS source,
        id(b) AS target
      '
    )
    YIELD graphName, nodeCount, relationshipCount
    RETURN graphName, nodeCount, relationshipCount
    """

    with driver.session() as session:
        rec = session.run(query, timeout=TIMEOUT).single()
        print("[OK] GDS graph created:", dict(rec))


def compute_degree(driver):
    print("[INFO] Computing Degree...")

    query = """
    CALL gds.degree.write(
      'address_graph',
      {
        writeProperty: 'degree',
        orientation: 'UNDIRECTED'
      }
    )
    YIELD nodePropertiesWritten
    RETURN nodePropertiesWritten
    """

    with driver.session() as session:
        result = session.run(query, timeout=TIMEOUT).single()
        print("[INFO] Degree written:", result["nodePropertiesWritten"])


def compute_pagerank(driver):
    print("[INFO] Computing PageRank...")

    query = """
    CALL gds.pageRank.write(
      'address_graph',
      {
        writeProperty: 'pageRank',
        maxIterations: 20,
        dampingFactor: 0.85
      }
    )
    YIELD nodePropertiesWritten, ranIterations
    RETURN nodePropertiesWritten, ranIterations
    """

    with driver.session() as session:
        result = session.run(query, timeout=TIMEOUT).single()
        print("[INFO] PageRank written:", result["nodePropertiesWritten"])
        print("[INFO] Ran iterations:", result["ranIterations"])


def compute_eigenvector(driver):
    print("[INFO] Computing Eigenvector Centrality...")

    query = """
    CALL gds.eigenvector.write(
      'address_graph',
      {
        writeProperty: 'eigenvector',
        maxIterations: 20
      }
    )
    YIELD nodePropertiesWritten, ranIterations
    RETURN nodePropertiesWritten, ranIterations
    """

    with driver.session() as session:
        result = session.run(query, timeout=TIMEOUT).single()
        print("[INFO] Eigenvector written:", result["nodePropertiesWritten"])
        print("[INFO] Ran iterations:", result["ranIterations"])


def run_gds_centrality(driver):
    ensure_address_graph_id(driver)
    project_graph(driver)
    compute_degree(driver)
    compute_pagerank(driver)

    if RUN_EIGENVECTOR:
        compute_eigenvector(driver)


def create_transaction_buffer():
    return {
        "graph_id": [],
        "date": [],
        "from": [],
        "value": [],
        "sender_degree": [],
        "sender_pageRank": [],
        "sender_eigenvector": [],
        "receiver_degree": [],
        "receiver_pageRank": [],
        "receiver_eigenvector": [],
    }


def save_transaction_chunk(buf, chunk_idx):
    df = pd.DataFrame(buf)

    df["graph_id"] = pd.to_numeric(df["graph_id"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    df["date"] = pd.to_datetime(
        df["date"],
        format=DATE_FMT,
        errors="coerce",
        exact=True,
    )

    df = df.dropna(subset=["graph_id", "date", "from"])
    df["graph_id"] = df["graph_id"].astype(np.int64)
    df["from"] = df["from"].astype("string")

    df["value"] = np.log1p(df["value"].clip(lower=0)).astype(np.float32)

    for col in CENTRALITY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

    df = df.sort_values(["date", "graph_id"]).reset_index(drop=True)

    out = os.path.join(OUTPUT_DIR, f"tx_chunk_{chunk_idx:04d}.parquet")
    df.to_parquet(out, engine="fastparquet", index=False)

    print(f"[INFO] wrote {out} rows={len(df)}")


def get_graph_id_range(driver):
    query = """
    MATCH (t:RawTransaction)
    RETURN min(t.graph_id) AS min_id, max(t.graph_id) AS max_id
    """

    with driver.session() as session:
        rec = session.run(query, timeout=TIMEOUT).single()
        return int(rec["min_id"]), int(rec["max_id"])


def export_transaction_chunks(driver):
    print("[INFO] Exporting RawTransaction ordered by from, date, graph_id...")

    ROWS_PER_CHUNK = 250_000
    RANGE_SIZE = 250_000

    minmax_query = """
        MATCH (t:RawTransaction)
        RETURN min(t.graph_id) AS min_id, max(t.graph_id) AS max_id
        """

    export_query = """
    MATCH (t:RawTransaction)
    WHERE t.graph_id >= $start_id
      AND t.graph_id < $end_id

    OPTIONAL MATCH (tx:RawTransaction {id: t.id})
    OPTIONAL MATCH (sender:Address)-[:SENT]->(tx)
    WITH t, tx, head(collect(sender)) AS sender

    OPTIONAL MATCH (tx)-[:RECEIVED]->(receiver:Address)
    WITH t, sender, head(collect(receiver)) AS receiver

    RETURN
        t.graph_id AS graph_id,
        t.date AS date,
        t.from AS `from`,
        t.value AS value,

        coalesce(sender.degree, 0.0) AS sender_degree,
        coalesce(sender.pageRank, 0.0) AS sender_pageRank,
        coalesce(sender.eigenvector, 0.0) AS sender_eigenvector,

        coalesce(receiver.degree, 0.0) AS receiver_degree,
        coalesce(receiver.pageRank, 0.0) AS receiver_pageRank,
        coalesce(receiver.eigenvector, 0.0) AS receiver_eigenvector
    """

    chunk_idx = 0
    buf = create_transaction_buffer()

    with driver.session() as session:
        mm = session.run(minmax_query, timeout=TIMEOUT).single()
        min_id = int(mm["min_id"])
        max_id = int(mm["max_id"])

        print(f"[INFO] graph_id range: {min_id} - {max_id}")

        start_id = min_id

        while start_id <= max_id:
            end_id = start_id + RANGE_SIZE

            print(f"[INFO] Exporting graph_id [{start_id}, {end_id})")

            result = session.run(
                export_query,
                start_id=start_id,
                end_id=end_id,
                timeout=TIMEOUT,
                fetch_size=5000
            )

            row_count = 0

            for rec in result:
                for k in buf:
                    buf[k].append(rec[k])

                row_count += 1

                if len(buf["graph_id"]) >= ROWS_PER_CHUNK:
                    save_transaction_chunk(buf, chunk_idx)
                    print(f"[OK] wrote chunk {chunk_idx}")
                    chunk_idx += 1
                    buf = create_transaction_buffer()

            print(f"[INFO] Range completed: rows={row_count}")

            start_id = end_id

        if buf["graph_id"]:
            save_transaction_chunk(buf, chunk_idx)
            print(f"[OK] wrote final chunk {chunk_idx}")

    print("[OK] Export completed")


def sort_transaction_chunks():
    print("[INFO] Sorting tx_chunk parquet files by date, graph_id using DuckDB...")

    input_pattern = os.path.join(OUTPUT_DIR, "tx_chunk_*.parquet")
    sorted_dir = os.path.join(OUTPUT_DIR, "sorted_chunks")
    os.makedirs(sorted_dir, exist_ok=True)

    con = duckdb.connect()
    temp_sorted = os.path.join(sorted_dir, "tx_sorted.parquet")

    query = f"""
    COPY (
        SELECT *
        FROM read_parquet('{input_pattern}')
        ORDER BY date, graph_id
    )
    TO '{temp_sorted}'
    (FORMAT PARQUET);
    """

    con.execute(query)
    con.close()

    print("[INFO] Splitting sorted parquet into chunks...")

    df_all = pd.read_parquet(temp_sorted, engine="fastparquet")
    chunk_size = 250_000

    for idx, start in enumerate(range(0, len(df_all), chunk_size)):
        chunk = df_all.iloc[start:start + chunk_size]
        out_path = os.path.join(sorted_dir, f"stx_chunk_{idx:04d}.parquet")
        chunk.to_parquet(out_path, engine="fastparquet", index=False)

        print(
            f"[INFO] wrote {out_path} rows={len(chunk)} "
            f"date_min={chunk['date'].min()} date_max={chunk['date'].max()}"
        )

    os.remove(temp_sorted)
    print("[OK] Sorted parquet chunks ready")


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "value",
        "seconds_since_last",
        "tx_count_15min",
        "tx_count_60min",
        "tx_count_15min_delta",
        "tx_count_15min_z",
        "value_std_15m",
        "active_sender_15m",
        "value_std_15m_log",
        "active_sender_15m_log",
        "same_timestamp_ratio_15m",
        "same_timestamp_count",
        *CENTRALITY_COLUMNS,
    ]

    for c in numeric_cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

    for c in CENTRALITY_COLUMNS:
        out[c + "_log"] = np.log1p(out[c].clip(lower=0)).astype(np.float32)

    out["max_degree_log"] = out[["sender_degree_log", "receiver_degree_log"]].max(axis=1).astype(np.float32)
    out["max_pageRank_log"] = out[["sender_pageRank_log", "receiver_pageRank_log"]].max(axis=1).astype(np.float32)
    out["max_eigenvector_log"] = out[["sender_eigenvector_log", "receiver_eigenvector_log"]].max(axis=1).astype(
        np.float32)

    out["seconds_since_last_log"] = np.log1p(out["seconds_since_last"].clip(lower=0)).astype(np.float32)
    out["tx_count_15min_log"] = np.log1p(out["tx_count_15min"].clip(lower=0)).astype(np.float32)
    out["tx_count_60min_log"] = np.log1p(out["tx_count_60min"].clip(lower=0)).astype(np.float32)
    out["tx_count_15min_delta_abs_log"] = np.log1p(out["tx_count_15min_delta"].abs()).astype(np.float32)

    out["tx_count_15min_z"] = out["tx_count_15min_z"].astype(np.float32)
    out["value_std_15m_log"] = np.log1p(out["value_std_15m"].clip(lower=0)).astype(np.float32)
    out["active_sender_15m_log"] = np.log1p(out["active_sender_15m"].clip(lower=0)).astype(np.float32)
    return out


def build_feature_chunks():
    print("[INFO] Feature engineering parquet chunks with market-aligned global buckets...")

    sorted_dir = os.path.join(INPUT_DIR, "sorted_chunks")

    files = sorted([
        f for f in os.listdir(sorted_dir)
        if f.startswith("stx_chunk_") and f.endswith(".parquet")
    ])

    if not files:
        raise RuntimeError("No stx_chunk parquet files found.")

    parts = []

    for fn in files:
        path = os.path.join(sorted_dir, fn)
        df = pd.read_parquet(path, engine="fastparquet")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df = df.dropna(subset=["date", "graph_id", "from"])
        parts.append(df)

    df_all = pd.concat(parts, ignore_index=True)
    df_all = df_all.sort_values(["date", "graph_id"]).reset_index(drop=True)

    print("[INFO] Total rows for feature engineering:", len(df_all))

    df_all["bucket_15m"] = df_all["date"].dt.floor("15min")
    df_all["bucket_60m"] = df_all["date"].dt.floor("60min")

    df_all["same_timestamp_count"] = (
        df_all.groupby("date")["graph_id"]
        .transform("count")
        .astype(np.float32)
    )

    df_all["same_timestamp_ratio_15m"] = (
            df_all["same_timestamp_count"] /
            (
                df_all.groupby("bucket_15m")["graph_id"]
                .transform("count")
                .clip(lower=1)
            )
    ).astype(np.float32)

    count15 = df_all.groupby("bucket_15m").size().sort_index()
    count60 = df_all.groupby("bucket_60m").size().sort_index()

    value_std_15m = (
        df_all.groupby("bucket_15m")["value"]
        .std()
        .sort_index()
        .fillna(0)
    )

    active_sender_15m = (
        df_all.groupby("bucket_15m")["from"]
        .nunique()
        .sort_index()
    )

    value_std_15m.name = "value_std_15m"
    active_sender_15m.name = "active_sender_15m"

    count15.name = "tx_count_15min"
    count60.name = "tx_count_60min"

    delta15 = count15.diff().fillna(0)
    delta15.name = "tx_count_15min_delta"

    rolling_mean = count15.shift(1).rolling(
        window=96,
        min_periods=12
    ).mean()

    rolling_std = count15.shift(1).rolling(
        window=96,
        min_periods=12
    ).std(ddof=0)

    adaptive_floor = rolling_mean * 0.10

    rolling_std = np.maximum(
        rolling_std,
        adaptive_floor
    )

    rolling_std = rolling_std.clip(lower=10000.0)

    z15 = (count15 - rolling_mean) / rolling_std

    z15 = (
        z15.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    z15.name = "tx_count_15min_z"

    df_all = df_all.merge(count15, left_on="bucket_15m", right_index=True, how="left")
    df_all = df_all.merge(delta15, left_on="bucket_15m", right_index=True, how="left")
    df_all = df_all.merge(z15, left_on="bucket_15m", right_index=True, how="left")
    df_all = df_all.merge(count60, left_on="bucket_60m", right_index=True, how="left")
    df_all = df_all.merge(value_std_15m, left_on="bucket_15m", right_index=True, how="left")
    df_all = df_all.merge(active_sender_15m, left_on="bucket_15m", right_index=True, how="left")

    df_all = df_all.sort_values(["from", "date", "graph_id"]).reset_index(drop=True)

    df_all["seconds_since_last"] = (
        df_all.groupby("from")["date"]
        .diff()
        .dt.total_seconds()
    )

    df_all["seconds_since_last"] = (
        df_all["seconds_since_last"]
        .fillna(86400)
        .clip(lower=1)
        .astype(np.float32)
    )

    df_all = df_all.sort_values(["date", "graph_id"]).reset_index(drop=True)

    for col in [
        "tx_count_15min",
        "tx_count_60min",
        "tx_count_15min_delta",
        "tx_count_15min_z",
        "value_std_15m",
        "active_sender_15m",
    ]:
        df_all[col] = (
            pd.to_numeric(df_all[col], errors="coerce")
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .astype(np.float32)
        )

    for col in CENTRALITY_COLUMNS:
        if col not in df_all.columns:
            df_all[col] = 0.0
        df_all[col] = (
            pd.to_numeric(df_all[col], errors="coerce")
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .astype(np.float32)
        )

    df_all = df_all.drop(columns=["bucket_15m", "bucket_60m"], errors="ignore")
    df_all = add_model_features(df_all)

    old_fe = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("fe_chunk_") and f.endswith(".parquet")
    ]

    for f in old_fe:
        os.remove(os.path.join(OUTPUT_DIR, f))

    chunk_size = 250_000

    for fe_idx, start in enumerate(range(0, len(df_all), chunk_size)):
        chunk = df_all.iloc[start:start + chunk_size].copy()
        out = os.path.join(OUTPUT_DIR, f"fe_chunk_{fe_idx:04d}.parquet")
        chunk.to_parquet(out, engine="fastparquet", index=False)

        print(
            f"[INFO] wrote {out} rows={len(chunk)} "
            f"date_min={chunk['date'].min()} "
            f"date_max={chunk['date'].max()} "
            f"max15={chunk['tx_count_15min'].max()} "
            f"max60={chunk['tx_count_60min'].max()} "
            f"min_z15={chunk['tx_count_15min_z'].min()} "
            f"max_z15={chunk['tx_count_15min_z'].max()}"
            f"max_value_std={chunk['value_std_15m'].max()} "
            f"max_active_sender={chunk['active_sender_15m'].max()}"
        )

    print("[OK] Market-aligned feature engineering completed")


def compute_global_thresholds(fe_files):
    print("[INFO] Computing global thresholds...")

    cols = [
        "date",
        "tx_count_15min",
        "tx_count_60min",
        "tx_count_15min_delta",
        "seconds_since_last",
        "tx_count_15min_z",
        "max_degree_log",
        "max_pageRank_log",
        "max_eigenvector_log",
        "value_std_15m_log",
        "active_sender_15m_log",
        "same_timestamp_ratio_15m",
        "same_timestamp_count"
    ]

    parts = []

    for fn in fe_files:
        df = pd.read_parquet(
            os.path.join(FEATURE_INPUT_DIR, fn),
            columns=cols,
            engine="fastparquet",
        )
        parts.append(df)

    big = pd.concat(parts, ignore_index=True)
    big = big.replace([np.inf, -np.inf], 0).fillna(0)
    big["date"] = pd.to_datetime(big["date"], errors="coerce")
    big = big[big["date"] < pd.Timestamp(TRAIN_END)]

    for c in cols:
        big[c] = pd.to_numeric(big[c], errors="coerce").fillna(0)

    nonzero_ssl = big["seconds_since_last"].replace(0, np.nan).dropna()

    delta = big["tx_count_15min_delta"]
    delta_pos = delta[delta > 0]
    delta_neg = delta[delta < 0].abs()

    thresholds = {
        "tx15_q50": float(big["tx_count_15min"].quantile(0.50)),
        "tx15_q90": float(big["tx_count_15min"].quantile(0.90)),
        "tx15_q95": float(big["tx_count_15min"].quantile(0.95)),
        "tx15_q99": float(big["tx_count_15min"].quantile(0.99)),
        "tx15_q995": float(big["tx_count_15min"].quantile(0.995)),

        "tx60_q50": float(big["tx_count_60min"].quantile(0.50)),
        "tx60_q90": float(big["tx_count_60min"].quantile(0.90)),
        "tx60_q95": float(big["tx_count_60min"].quantile(0.95)),
        "tx60_q99": float(big["tx_count_60min"].quantile(0.99)),
        "tx60_q995": float(big["tx_count_60min"].quantile(0.995)),

        "z15_q85": float(big["tx_count_15min_z"].quantile(0.85)),
        "z15_q90": float(big["tx_count_15min_z"].quantile(0.90)),
        "z15_q95": float(big["tx_count_15min_z"].quantile(0.95)),
        "z15_q99": float(big["tx_count_15min_z"].quantile(0.99)),
        "z15_q995": float(big["tx_count_15min_z"].quantile(0.995)),

        "delta15_abs_q85": float(delta.abs().quantile(0.85)),
        "delta15_abs_q90": float(delta.abs().quantile(0.90)),
        "delta15_abs_q95": float(delta.abs().quantile(0.95)),
        "delta15_abs_q99": float(delta.abs().quantile(0.99)),
        "delta15_abs_q995": float(delta.abs().quantile(0.995)),

        "delta15_pos_q95": float(delta_pos.quantile(0.95)) if len(delta_pos) > 0 else 0.0,
        "delta15_pos_q99": float(delta_pos.quantile(0.99)) if len(delta_pos) > 0 else 0.0,

        "delta15_neg_abs_q95": float(delta_neg.quantile(0.95)) if len(delta_neg) > 0 else 0.0,
        "delta15_neg_abs_q99": float(delta_neg.quantile(0.99)) if len(delta_neg) > 0 else 0.0,

        "delta_std": float(delta.std(ddof=0)),

        "sec_q20": float(nonzero_ssl.quantile(0.20)) if len(nonzero_ssl) > 0 else 0.0,
        "sec_q10": float(nonzero_ssl.quantile(0.10)) if len(nonzero_ssl) > 0 else 0.0,
        "sec_q05": float(nonzero_ssl.quantile(0.05)) if len(nonzero_ssl) > 0 else 0.0,
        "sec_q30": float(nonzero_ssl.quantile(0.30)) if len(nonzero_ssl) > 0 else 0.0,

        "max_degree_log_q99": float(big["max_degree_log"].quantile(0.99)),
        "max_degree_log_q90": float(big["max_degree_log"].quantile(0.90)),
        "max_degree_log_q95": float(big["max_degree_log"].quantile(0.95)),
        "max_degree_log_q995": float(big["max_degree_log"].quantile(0.995)),
        "max_pageRank_log_q99": float(big["max_pageRank_log"].quantile(0.99)),
        "max_pageRank_log_q90": float(big["max_pageRank_log"].quantile(0.90)),
        "max_eigenvector_log_q99": float(big["max_eigenvector_log"].quantile(0.99)),
        "max_eigenvector_log_q90": float(big["max_eigenvector_log"].quantile(0.90)),

        "value_std_15m_log_q85": float(big["value_std_15m_log"].quantile(0.85)),
        "value_std_15m_log_q90": float(big["value_std_15m_log"].quantile(0.90)),
        "value_std_15m_log_q95": float(big["value_std_15m_log"].quantile(0.95)),
        "value_std_15m_log_q99": float(big["value_std_15m_log"].quantile(0.99)),
        "value_std_15m_log_q995": float(big["value_std_15m_log"].quantile(0.995)),

        "active_sender_15m_log_q85": float(big["active_sender_15m_log"].quantile(0.85)),
        "active_sender_15m_log_q90": float(big["active_sender_15m_log"].quantile(0.90)),
        "active_sender_15m_log_q95": float(big["active_sender_15m_log"].quantile(0.95)),
        "active_sender_15m_log_q99": float(big["active_sender_15m_log"].quantile(0.99)),
        "active_sender_15m_log_q995": float(big["active_sender_15m_log"].quantile(0.995)),

        "same_timestamp_count_q90": float(big["same_timestamp_count"].quantile(0.50)),

    }

    print("[OK] global thresholds:", thresholds)

    pd.DataFrame([thresholds]).to_csv(
        os.path.join(OUTPUT_DIR, "global_thresholds.csv"),
        index=False
    )

    del big, parts
    return thresholds


def train_isolation_forest(fe_files):
    print("[INFO] Training IsolationForest...")

    rng = np.random.RandomState(IF_RANDOM_STATE)
    sample_parts = []
    collected = 0

    for fn in fe_files:
        if collected >= IF_FIT_SAMPLE:
            break

        df = pd.read_parquet(
            os.path.join(FEATURE_INPUT_DIR, fn),
            columns=["date", *MODEL_FEATURE_COLUMNS],
            engine="fastparquet",
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] < pd.Timestamp(TRAIN_END)]
        df = df[MODEL_FEATURE_COLUMNS]

        remaining = IF_FIT_SAMPLE - collected
        take = min(len(df), max(50_000, remaining // 4))

        if len(df) > take:
            idx = rng.choice(len(df), size=take, replace=False)
        else:
            idx = np.arange(len(df))

        sample_parts.append(df.iloc[idx])
        collected += len(idx)

    X_fit = (
        pd.concat(sample_parts, ignore_index=True)
        .astype(np.float32)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    scaler = RobustScaler()
    X_fit_scaled = scaler.fit_transform(X_fit).astype(np.float64)

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=1,
    )

    model.fit(X_fit_scaled)

    # Calibrate the threshold on training observations only.

    train_scores = -model.decision_function(X_fit_scaled)
    train_threshold = float(
        np.percentile(train_scores, 100.0 * (1.0 - IF_CONTAMINATION))
    )

    print(
        f"[OK] IF trained. fit_rows={len(X_fit_scaled)} "
        f"train_threshold={train_threshold:.6f}"
    )

    if HAS_JOBLIB:
        joblib.dump(model, os.path.join(OUTPUT_DIR, "if_model.joblib"))
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, "if_scaler.joblib"))

    pd.DataFrame([{
        "threshold": train_threshold,
        "contamination": IF_CONTAMINATION
    }]).to_csv(
        os.path.join(OUTPUT_DIR, "if_train_threshold.csv"),
        index=False
    )

    return model, scaler, train_threshold


def split_by_period(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    train_mask = df["date"] < pd.Timestamp(TRAIN_END)

    valid_mask = (
            (df["date"] >= pd.Timestamp(TRAIN_END)) &
            (df["date"] < pd.Timestamp(VALID_END))
    )

    test_mask = (
            df["date"] >= pd.Timestamp(TEST_START)
    )

    return train_mask, valid_mask, test_mask


def score_isolation_forest(fe_files, model, scaler, thresholds, if_threshold):
    """Score all chunks using an IF threshold learned from TRAIN only."""
    print("[INFO] Scoring IsolationForest with TRAIN-only anomaly threshold...")
    print(f"[INFO] IF train_threshold={if_threshold}")

    out_idx = 0

    for fn in fe_files:
        path = os.path.join(FEATURE_INPUT_DIR, fn)
        df = pd.read_parquet(path, engine="fastparquet")
        df = add_model_features(df)

        X = (
            df[MODEL_FEATURE_COLUMNS]
            .astype(np.float32)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        Xs = scaler.transform(X).astype(np.float32)
        score = -model.decision_function(Xs)

        df[IF_SCORE_COLUMN] = score.astype(np.float32)
        df[IF_LABEL_COLUMN] = (score >= if_threshold).astype(np.int8)

        train_mask, valid_mask, test_mask = split_by_period(df)

        df["dataset_split"] = "TRAIN"
        df.loc[valid_mask, "dataset_split"] = "VALID"
        df.loc[test_mask, "dataset_split"] = "TEST"

        market_density_flag = (
                (df["tx_count_15min"] >= thresholds["tx15_q95"]) &
                (df["tx_count_60min"] >= thresholds["tx60_q95"])
        )

        market_spike_flag = (
                (df["tx_count_15min_z"] >= thresholds["z15_q90"]) &
                (df["tx_count_15min_delta"].abs() >= thresholds["delta15_abs_q90"])
        )

        market_volatility_flag = (
                (df["value_std_15m"] >= thresholds["value_std_15m_log_q95"]) |
                (df["active_sender_15m"] >= thresholds["active_sender_15m_log_q95"])
        )

        rapid_sender_flag = (
                                    df["seconds_since_last"] <= thresholds["sec_q10"]
                            ) & (
                                    df["active_sender_15m"] >= thresholds["active_sender_15m_log_q95"]
                            ) & (
                                    df["tx_count_15min_z"] >= thresholds["z15_q90"]
                            )

        df["time_anomaly_flag"] = (
                (
                        market_spike_flag
                        &
                        market_volatility_flag
                )
                |
                (
                        market_density_flag
                        &
                        rapid_sender_flag
                        &
                        (df[IF_LABEL_COLUMN] == 1)
                )
        ).astype(np.int8)

        high_graph = (
                (df["max_degree_log"] >= thresholds["max_degree_log_q995"]) |
                (
                        (df["max_pageRank_log"] >= thresholds["max_pageRank_log_q99"]) &
                        (df["max_eigenvector_log"] >= thresholds["max_eigenvector_log_q99"])
                )
        )

        df["graph_anomaly_flag"] = (
                high_graph &
                (
                        (df[IF_LABEL_COLUMN] == 1) |
                        (df["time_anomaly_flag"] == 1)
                )
        ).astype(np.int8)

        df["final_anomaly"] = df[IF_LABEL_COLUMN].astype(np.int8)

        df["anomaly_label_market"] = (
                market_volatility_flag |
                market_spike_flag
        ).astype(np.int8)

        out = os.path.join(OUTPUT_DIR, f"base_scored_chunk_{out_idx:04d}.parquet")
        df.to_parquet(out, engine="fastparquet", index=False)

        for split_name in ["TRAIN", "VALID", "TEST"]:
            sdf = df[df["dataset_split"] == split_name]

            if len(sdf) == 0:
                continue

            print(
                f"[{split_name}] "
                f"rows={len(sdf)} "
                f"if={int(sdf[IF_LABEL_COLUMN].sum())} "
                f"time={int(sdf['time_anomaly_flag'].sum())} "
                f"graph={int(sdf['graph_anomaly_flag'].sum())} "
                f"final={int(sdf['final_anomaly'].sum())}"
            )

        out_idx += 1

    return float(if_threshold)


def _equal_day_quotas(days, total_rows):
    """Distribute a requested sample size as evenly as possible across TRAIN days."""
    if not days:
        raise RuntimeError("No TRAIN days found.")

    base = total_rows // len(days)
    rem = total_rows % len(days)

    return {
        day: base + (1 if i < rem else 0)
        for i, day in enumerate(days)
    }


def _sample_training_days(
        fe_files,
        fit_rows,
        calibration_rows,
        random_state,
):
    print(
        f"[INFO] Building temporal-stratified TRAIN samples: "
        f"fit={fit_rows}, calibration={calibration_rows}"
    )

    train_end = pd.Timestamp(TRAIN_END)

    day_counts = {}
    for fn in fe_files:
        df_date = pd.read_parquet(
            os.path.join(FEATURE_INPUT_DIR, fn),
            columns=["date"],
            engine="fastparquet",
        )
        d = pd.to_datetime(df_date["date"], errors="coerce")
        d = d[d < train_end].dt.floor("D")

        vc = d.value_counts()
        for day, cnt in vc.items():
            day_counts[day] = day_counts.get(day, 0) + int(cnt)

    days = sorted(day_counts)
    if not days:
        raise RuntimeError("No TRAIN rows found for temporal-stratified sampling.")

    fit_quota = _equal_day_quotas(days, fit_rows)
    cal_quota = _equal_day_quotas(days, calibration_rows)
    total_quota = {
        day: fit_quota[day] + cal_quota[day]
        for day in days
    }

    too_small = [
        (day, day_counts[day], total_quota[day])
        for day in days
        if day_counts[day] < total_quota[day]
    ]
    if too_small:
        raise RuntimeError(
            "Some TRAIN days do not contain enough rows for the requested "
            f"fit+calibration quotas: {too_small}"
        )

    rng = np.random.RandomState(random_state)
    reservoirs = {day: None for day in days}

    cols = ["date", *MODEL_FEATURE_COLUMNS]

    for fn in fe_files:
        df = pd.read_parquet(
            os.path.join(FEATURE_INPUT_DIR, fn),
            columns=cols,
            engine="fastparquet",
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] < train_end].copy()
        if df.empty:
            continue

        df["_sample_day"] = df["date"].dt.floor("D")

        for day, g in df.groupby("_sample_day", sort=False):
            if day not in total_quota:
                continue

            q = total_quota[day]
            candidate = g[MODEL_FEATURE_COLUMNS].copy()
            candidate["_rand_key"] = rng.random(len(candidate))

            prev = reservoirs[day]
            if prev is not None:
                candidate = pd.concat([prev, candidate], ignore_index=True)

            if len(candidate) > q:
                candidate = candidate.nsmallest(q, "_rand_key")

            reservoirs[day] = candidate

    fit_parts = []
    cal_parts = []
    diag_rows = []

    for i, day in enumerate(days):
        r = reservoirs[day]
        q_fit = fit_quota[day]
        q_cal = cal_quota[day]

        if r is None or len(r) < (q_fit + q_cal):
            raise RuntimeError(
                f"Reservoir for {day.date()} has {0 if r is None else len(r)} rows, "
                f"but {q_fit + q_cal} are required."
            )

        r = r.sample(
            frac=1.0,
            random_state=random_state + i + 1
        ).reset_index(drop=True)

        fit_day = r.iloc[:q_fit][MODEL_FEATURE_COLUMNS]
        cal_day = r.iloc[q_fit:q_fit + q_cal][MODEL_FEATURE_COLUMNS]

        fit_parts.append(fit_day)
        cal_parts.append(cal_day)

        diag_rows.append({
            "day": day,
            "train_rows": day_counts[day],
            "fit_rows": len(fit_day),
            "calibration_rows": len(cal_day),
        })

    X_fit = (
        pd.concat(fit_parts, ignore_index=True)
        .astype(np.float32)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    X_cal = (
        pd.concat(cal_parts, ignore_index=True)
        .astype(np.float32)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(
        os.path.join(OUTPUT_DIR, "baseline_sampling_by_day.csv"),
        index=False,
    )

    print("[OK] Temporal-stratified sampling completed.")
    print(diag.to_string(index=False))
    print(f"[OK] fit_rows={len(X_fit)}, calibration_rows={len(X_cal)}")

    return X_fit, X_cal


def _detector_anomaly_score(name, model, Xs):
    """Return scores with a common direction: larger value = more anomalous."""
    if name == "IF":
        return -model.decision_function(Xs)
    if name == "LOF":
        return -model.decision_function(Xs)
    if name == "OCSVM":
        return -model.decision_function(Xs)
    if name in ("COPOD", "ECOD"):
        return model.decision_function(Xs)
    raise ValueError(f"Unknown baseline model: {name}")


def train_benchmark_detectors(fe_files):
    print(
        "[INFO] Training fair temporal-stratified benchmark: "
        "IF, LOF, OCSVM, COPOD, ECOD..."
    )

    X_fit, X_cal = _sample_training_days(
        fe_files=fe_files,
        fit_rows=BASELINE_FIT_SAMPLE,
        calibration_rows=BASELINE_CALIBRATION_SAMPLE,
        random_state=BASELINE_RANDOM_STATE,
    )

    scaler = RobustScaler()
    X_fit_scaled = scaler.fit_transform(X_fit).astype(np.float64)
    X_cal_scaled = scaler.transform(X_cal).astype(np.float64)

    models = {
        "IF": IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=BASELINE_CONTAMINATION,
            random_state=BASELINE_RANDOM_STATE,
            n_jobs=1,
        ),
        "LOF": LocalOutlierFactor(
            n_neighbors=LOF_N_NEIGHBORS,
            contamination=BASELINE_CONTAMINATION,
            novelty=True,
            n_jobs=1,
        ),
        "OCSVM": OneClassSVM(
            kernel=OCSVM_KERNEL,
            gamma=OCSVM_GAMMA,
            nu=OCSVM_NU,
        ),
    }

    if HAS_PYOD:
        models["COPOD"] = COPOD(contamination=BASELINE_CONTAMINATION)
        models["ECOD"] = ECOD(contamination=BASELINE_CONTAMINATION)
    else:
        print("[WARN] pyod not installed (pip install pyod). COPOD/ECOD will be skipped.")
        models["COPOD"] = None
        models["ECOD"] = None

    calibration_thresholds = {}
    runtime_rows = []

    if HAS_JOBLIB:
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, "baseline_scaler.joblib"))

    for name in BENCHMARK_DETECTORS:
        model = models.get(name)
        if model is None:
            calibration_thresholds[name] = None
            continue

        print(
            f"[INFO] Fitting benchmark {name} on "
            f"{len(X_fit_scaled)} temporal-stratified TRAIN rows..."
        )

        t0 = time.perf_counter()
        model.fit(X_fit_scaled)
        fit_seconds = time.perf_counter() - t0

        # Calibrate on a separate, disjoint training sample.

        cal_score = np.asarray(
            _detector_anomaly_score(name, model, X_cal_scaled),
            dtype=np.float64,
        )
        threshold = float(
            np.percentile(
                cal_score,
                100.0 * (1.0 - BASELINE_CONTAMINATION)
            )
        )

        if name == "OCSVM":
            print(
                "[OCSVM DIAG]",
                f"min={cal_score.min():.12f}",
                f"p50={np.percentile(cal_score, 50):.12f}",
                f"p95={np.percentile(cal_score, 95):.12f}",
                f"p98.5={np.percentile(cal_score, 98.5):.12f}",
                f"p99={np.percentile(cal_score, 99):.12f}",
                f"max={cal_score.max():.12f}",
                f"unique={np.unique(cal_score).size}",
                f"gt={100 * np.mean(cal_score > threshold):.4f}%",
                f"eq={100 * np.mean(cal_score == threshold):.4f}%",
                f"ge={100 * np.mean(cal_score >= threshold):.4f}%"
            )

        calibration_thresholds[name] = threshold

        cal_anomalies = int((cal_score > threshold).sum())
        cal_rate = 100.0 * cal_anomalies / len(cal_score)

        runtime_rows.append({
            "model": name,
            "fit_rows": len(X_fit_scaled),
            "calibration_rows": len(X_cal_scaled),
            "fit_seconds": fit_seconds,
            "calibration_threshold": threshold,
            "calibration_anomalies": cal_anomalies,
            "calibration_anomaly_rate_pct": cal_rate,
            "contamination": BASELINE_CONTAMINATION,
        })

        if HAS_JOBLIB:
            joblib.dump(
                model,
                os.path.join(OUTPUT_DIR, f"baseline_{name.lower()}_model.joblib")
            )

        print(
            f"[OK] {name} trained. "
            f"fit_rows={len(X_fit_scaled)} "
            f"calibration_rows={len(X_cal_scaled)} "
            f"fit_seconds={fit_seconds:.3f} "
            f"threshold={threshold:.6f} "
            f"calibration_rate={cal_rate:.3f}%"
        )

    threshold_df = pd.DataFrame([
        {
            "model": name,
            "threshold": calibration_thresholds.get(name),
            "contamination": BASELINE_CONTAMINATION,
            "fit_rows": len(X_fit_scaled),
            "calibration_rows": len(X_cal_scaled),
        }
        for name in BENCHMARK_DETECTORS
        if calibration_thresholds.get(name) is not None
    ])
    threshold_df.to_csv(
        os.path.join(OUTPUT_DIR, "baseline_thresholds.csv"),
        index=False,
    )

    pd.DataFrame(runtime_rows).to_csv(
        os.path.join(OUTPUT_DIR, "baseline_runtime.csv"),
        index=False,
    )

    return models, scaler, calibration_thresholds


def score_benchmark_detectors(fe_files, models, scaler, train_thresholds):
    print("[INFO] Scoring baseline models on all chunks with TRAIN-only thresholds...")

    # Remove previous outputs to avoid mixing runs.
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("baseline_scored_chunk_") and f.endswith(".parquet"):
            os.remove(os.path.join(OUTPUT_DIR, f))

    score_seconds = {name: 0.0 for name in BENCHMARK_DETECTORS}

    for out_idx, fn in enumerate(fe_files):
        path = os.path.join(FEATURE_INPUT_DIR, fn)
        df = pd.read_parquet(path, engine="fastparquet")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        X = (
            df[MODEL_FEATURE_COLUMNS]
            .astype(np.float32)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
        Xs = scaler.transform(X).astype(np.float32)

        train_mask, valid_mask, test_mask = split_by_period(df)
        df["dataset_split"] = "TRAIN"
        df.loc[valid_mask, "dataset_split"] = "VALID"
        df.loc[test_mask, "dataset_split"] = "TEST"

        for name in BENCHMARK_DETECTORS:
            model = models.get(name)
            threshold = train_thresholds.get(name)
            if model is None or threshold is None:
                continue

            t0 = time.perf_counter()
            score = np.asarray(_detector_anomaly_score(name, model, Xs), dtype=np.float32)
            score_seconds[name] += time.perf_counter() - t0

            prefix = "benchmark_if" if name == "IF" else name.lower()
            df[f"{prefix}_score"] = score
            df[f"{prefix}_label"] = (score > threshold).astype(np.int8)

        proposed_path = os.path.join(OUTPUT_DIR, f"base_scored_chunk_{out_idx:04d}.parquet")
        if os.path.exists(proposed_path):
            proposed = pd.read_parquet(
                proposed_path,
                columns=["graph_id", IF_LABEL_COLUMN],
                engine="fastparquet",
            ).rename(columns={IF_LABEL_COLUMN: "proposed_if_label"})
            df = df.merge(proposed, on="graph_id", how="left", validate="one_to_one")
            df["proposed_if_label"] = df["proposed_if_label"].fillna(0).astype(np.int8)

        out = os.path.join(OUTPUT_DIR, f"baseline_scored_chunk_{out_idx:04d}.parquet")
        df.to_parquet(out, engine="fastparquet", index=False)

        chunk_counts = []
        for name in BENCHMARK_DETECTORS:
            label_col = "benchmark_if_label" if name == "IF" else f"{name.lower()}_label"
            if label_col in df.columns:
                chunk_counts.append(f"{name}={int(df[label_col].sum())}")

        print(
            f"[OK] baseline chunk={out_idx} rows={len(df)} "
            + " ".join(chunk_counts)
        )

    runtime_path = os.path.join(OUTPUT_DIR, "baseline_runtime.csv")
    if os.path.exists(runtime_path):
        runtime = pd.read_csv(runtime_path)
        runtime["score_seconds"] = runtime["model"].map(score_seconds).fillna(0.0)
        runtime["total_seconds"] = runtime["fit_seconds"] + runtime["score_seconds"]
        runtime.to_csv(runtime_path, index=False)

    print(f"[OK] baseline scoring done. {len(fe_files)} chunks written.")
    return score_seconds


def compare_benchmark_detectors(baseline_files):
    'Export anomaly rates and pairwise Jaccard agreement for the benchmark.'
    print("[INFO] Comparing fair benchmark anomaly detectors...")

    label_cols = {
        "IF": "benchmark_if_label",
        "LOF": "lof_label",
        "OCSVM": "ocsvm_label",
        "COPOD": "copod_label",
        "ECOD": "ecod_label",
    }

    counts = {(split, name): [0, 0] for split in ["TRAIN", "VALID", "TEST", "ALL"] for name in label_cols}
    pairs = [(a, b) for i, a in enumerate(label_cols) for b in list(label_cols)[i + 1:]]
    overlaps = {(split, a, b): [0, 0] for split in ["TRAIN", "VALID", "TEST", "ALL"] for a, b in pairs}
    proposed_overlap = {name: [0, 0] for name in label_cols}

    for fn in baseline_files:
        df = pd.read_parquet(os.path.join(OUTPUT_DIR, fn), engine="fastparquet")

        for split in ["TRAIN", "VALID", "TEST", "ALL"]:
            sdf = df if split == "ALL" else df[df["dataset_split"] == split]
            if len(sdf) == 0:
                continue

            for name, col in label_cols.items():
                if col not in sdf.columns:
                    continue
                counts[(split, name)][0] += len(sdf)
                counts[(split, name)][1] += int(sdf[col].sum())

            for a, b in pairs:
                ca, cb = label_cols[a], label_cols[b]
                if ca not in sdf.columns or cb not in sdf.columns:
                    continue
                aa = sdf[ca].to_numpy(dtype=bool)
                bb = sdf[cb].to_numpy(dtype=bool)
                overlaps[(split, a, b)][0] += int(np.logical_and(aa, bb).sum())
                overlaps[(split, a, b)][1] += int(np.logical_or(aa, bb).sum())

        if "proposed_if_label" in df.columns:
            pp = df["proposed_if_label"].to_numpy(dtype=bool)
            for name, col in label_cols.items():
                if col not in df.columns:
                    continue
                mm = df[col].to_numpy(dtype=bool)
                proposed_overlap[name][0] += int(np.logical_and(pp, mm).sum())
                proposed_overlap[name][1] += int(np.logical_or(pp, mm).sum())

    summary_rows = []
    for split in ["TRAIN", "VALID", "TEST", "ALL"]:
        for name in label_cols:
            rows, anomalies = counts[(split, name)]
            summary_rows.append({
                "split": split,
                "model": name,
                "rows": rows,
                "anomaly_count": anomalies,
                "anomaly_rate_pct": (100.0 * anomalies / rows) if rows else 0.0,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "baseline_model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)

    jaccard_rows = []
    for split in ["TRAIN", "VALID", "TEST", "ALL"]:
        for a, b in pairs:
            inter, union = overlaps[(split, a, b)]
            jaccard_rows.append({
                "split": split,
                "method_a": a,
                "method_b": b,
                "intersection": inter,
                "union": union,
                "jaccard": (inter / union) if union else 1.0,
            })

    jaccard_df = pd.DataFrame(jaccard_rows)
    jaccard_path = os.path.join(OUTPUT_DIR, "baseline_pairwise_jaccard.csv")
    jaccard_df.to_csv(jaccard_path, index=False)

    proposed_rows = []
    for name, (inter, union) in proposed_overlap.items():
        if union:
            proposed_rows.append({
                "model": name,
                "intersection_with_proposed_IF": inter,
                "union_with_proposed_IF": union,
                "jaccard_vs_proposed_IF": inter / union,
            })
    if proposed_rows:
        pd.DataFrame(proposed_rows).to_csv(
            os.path.join(OUTPUT_DIR, "baseline_vs_proposed_if_jaccard.csv"), index=False
        )

    print("[OK] Baseline comparison saved:", summary_path)
    print("[OK] Pairwise Jaccard saved:", jaccard_path)
    print(summary_df)
    print(jaccard_df)
    return summary_df, jaccard_df


def print_daily_anomaly_summary():
    print("\n[INFO] Daily anomaly summary...\n")

    scored_files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("base_scored_chunk_") and f.endswith(".parquet")
    ])

    if not scored_files:
        raise RuntimeError("No scored_chunk parquet files found.")

    parts = []

    for fn in scored_files:
        df = pd.read_parquet(
            os.path.join(OUTPUT_DIR, fn),
            columns=["date", "graph_id", "final_anomaly"],
            engine="fastparquet"
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        df["day"] = df["date"].dt.date

        daily = (
            df.groupby("day")
            .agg(
                total_tx=("graph_id", "count"),
                anomaly_count=("final_anomaly", "sum")
            )
            .reset_index()
        )

        parts.append(daily)

    daily_summary = pd.concat(parts, ignore_index=True)

    daily_summary = (
        daily_summary.groupby("day", as_index=False)
        .agg(
            total_tx=("total_tx", "sum"),
            anomaly_count=("anomaly_count", "sum")
        )
    )

    daily_summary["anomaly_rate_percent"] = (
            100.0 * daily_summary["anomaly_count"] / daily_summary["total_tx"]
    ).round(3)

    daily_summary = daily_summary.sort_values("day")

    print(daily_summary.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, "daily_anomaly_summary.csv")
    daily_summary.to_csv(out_path, index=False)

    print(f"\n[OK] Saved daily anomaly summary to: {out_path}")


def event_comparison_validation():
    print("[INFO] Event Comparison Validation")

    scored_files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    ])

    event_results = event_grounded_validation(
        scored_files=scored_files,
        event_calendar_path=os.path.join(FEATURE_INPUT_DIR, f"event_calendar.csv")
    )


def print_daily_taxonomy_distribution(
        input_pattern="scored_chunk_*.csv",
        count_output_file="daily_taxonomy_counts.csv",
        percentage_output_file="daily_taxonomy_percentages.csv"
):
    files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    ])

    if not files:
        print(f"[WARN] No files found: {input_pattern}")
        return pd.DataFrame(), pd.DataFrame()

    dfs = []

    for f in files:
        try:
            df = pd.read_parquet(
                os.path.join(OUTPUT_DIR, f),
                engine="fastparquet",
            )

            if "taxonomy_id" not in df.columns or "date" not in df.columns:
                print(f"[SKIP] Missing columns: {os.path.basename(f)}")
                continue

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "taxonomy_id"])

            df["day"] = df["date"].dt.date
            dfs.append(df[["day", "taxonomy_id"]])

        except Exception as e:
            print(f"[ERROR] {os.path.basename(f)} -> {e}")

    if not dfs:
        print("[WARN] No valid taxonomy data found.")
        return pd.DataFrame(), pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)

    daily = (
        all_df
        .groupby(["day", "taxonomy_id"])
        .size()
        .reset_index(name="count")
    )

    count_pivot = (
        daily
        .pivot(index="day", columns="taxonomy_id", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    for tax in ["T1", "T2", "T3", "T4", "T5"]:
        if tax not in count_pivot.columns:
            count_pivot[tax] = 0

    count_pivot["total"] = (
            count_pivot["T1"] +
            count_pivot["T2"] +
            count_pivot["T3"] +
            count_pivot["T4"] +
            count_pivot["T5"]
    )

    count_pivot = count_pivot[
        ["day", "T1", "T2", "T3", "T4", "T5", "T6", "total"]
    ].sort_values("day")

    percentage_pivot = count_pivot.copy()

    for tax in ["T1", "T2", "T3", "T4", "T5"]:
        percentage_pivot[tax] = (
                percentage_pivot[tax] / percentage_pivot["total"] * 100
        ).round(3)

    percentage_pivot = percentage_pivot[
        ["day", "T1", "T2", "T3", "T4", "T5", "T6", "total"]
    ]

    count_pivot.to_csv(count_output_file, index=False)
    percentage_pivot.to_csv(percentage_output_file, index=False)

    print(f"[OK] Saved counts: {count_output_file}")
    print(f"[OK] Saved percentages: {percentage_output_file}")

    return count_pivot, percentage_pivot


def print_final_taxonomy_summary():
    print("\n[INFO] Final taxonomy summary...\n")

    scored_files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    ])

    if not scored_files:
        raise RuntimeError("No scored_chunk parquet files found.")

    rows = []

    for fn in scored_files:
        df = pd.read_parquet(
            os.path.join(OUTPUT_DIR, fn),
            columns=["taxonomy_id", "final_anomaly"],
            engine="fastparquet",
        )

        df = df[df["final_anomaly"] == 1]

        if len(df) == 0:
            continue

        counts = df["taxonomy_id"].value_counts()

        for taxonomy_id, count in counts.items():
            rows.append({
                "taxonomy_id": taxonomy_id,
                "count": int(count),
            })

    summary = pd.DataFrame(rows)

    if summary.empty:
        print("[WARN] No anomalies found.")
        return

    summary = (
        summary.groupby("taxonomy_id", as_index=False)["count"]
        .sum()
    )

    wanted = ["T1", "T2", "T3", "T4", "T5"]

    summary = (
        pd.DataFrame({"taxonomy_id": wanted})
        .merge(summary, on="taxonomy_id", how="left")
        .fillna({"count": 0})
    )

    total = summary["count"].sum()

    summary["percentage"] = (
            100.0 * summary["count"] / total
    ).round(3)

    print(summary)

    print("\n[OK] Total anomaly count:", int(total))

    out_path = os.path.join(OUTPUT_DIR, "final_taxonomy_summary.csv")
    summary.to_csv(out_path, index=False)

    print(f"[OK] Saved taxonomy summary to: {out_path}")


def print_validation_queries():
    print("\nValidation Cypher queries:\n")

    print(f"""
MATCH (t:RawTransaction)
RETURN
  count(*) AS total,
  count(t.{IF_LABEL_COLUMN}) AS if_labeled,
  sum(t.{IF_LABEL_COLUMN}) AS if_anomaly,
  sum(t.time_anomaly_flag) AS time_anomaly,
  sum(t.graph_anomaly_flag) AS graph_anomaly,
  sum(t.final_label) AS final_anomaly;
""")

    print("""
MATCH (t:RawTransaction)
WHERE t.final_label = 1
WITH substring(t.date, 0, 10) AS day, t.taxonomy_id AS taxonomy_id, count(*) AS cnt
RETURN day, taxonomy_id, cnt
ORDER BY day, cnt DESC;
""")

    print("""
MATCH (t:RawTransaction)
WHERE t.final_label = 1
WITH substring(t.date, 0, 10) AS day, count(*) AS anomaly_count
RETURN day, anomaly_count
ORDER BY day;
""")


def event_grounded_validation(scored_files, event_calendar_path):
    events = load_event_calendar(event_calendar_path)

    results = []

    for _, event in events.iterrows():
        event_id = event["event_id"]
        event_name = event["event_name"]
        event_type = event["event_type"]
        event_time = event["event_datetime"]
        window_hours = event["window_hours"]
        expected = set(event["expected_taxonomy"])

        start_time = event_time - pd.Timedelta(hours=window_hours)
        end_time = event_time + pd.Timedelta(hours=window_hours)

        event_rows = []

        for fn in scored_files:
            df = pd.read_parquet(os.path.join(OUTPUT_DIR, fn))

            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            sub = df[
                (df["date"] >= start_time) &
                (df["date"] <= end_time)
                ]

            if len(sub) > 0:
                event_rows.append(sub)

        if not event_rows:
            results.append({
                "event_id": event_id,
                "event_name": event_name,
                "event_type": event_type,
                "event_datetime": event_time,
                "window_hours": window_hours,
                "expected_taxonomy": ",".join(expected),
                "total_tx": 0,
                "anomaly_tx": 0,
                "anomaly_rate": 0,
                "matched_anomaly_tx": 0,
                "taxonomy_match_rate": 0,
                "dominant_taxonomy": None
            })
            continue

        event_df = pd.concat(event_rows, ignore_index=True)

        total_tx = len(event_df)

        anomaly_df = event_df[
            (event_df["final_anomaly"] == 1) |
            (event_df[IF_LABEL_COLUMN] == 1)
            ].copy()

        anomaly_tx = len(anomaly_df)

        if anomaly_tx > 0:
            taxonomy_counts = anomaly_df["taxonomy_id"].value_counts()
            dominant_taxonomy = taxonomy_counts.index[0]

            matched_anomaly_tx = anomaly_df[
                anomaly_df["taxonomy_id"].isin(expected)
            ].shape[0]

            taxonomy_match_rate = round(
                100.0 * matched_anomaly_tx / anomaly_tx, 3
            )
        else:
            dominant_taxonomy = None
            matched_anomaly_tx = 0
            taxonomy_match_rate = 0

        anomaly_rate = round(100.0 * anomaly_tx / total_tx, 3)

        row = {
            "event_id": event_id,
            "event_name": event_name,
            "event_type": event_type,
            "event_datetime": event_time,
            "window_hours": window_hours,
            "expected_taxonomy": ",".join(expected),
            "total_tx": total_tx,
            "anomaly_tx": anomaly_tx,
            "anomaly_rate": anomaly_rate,
            "matched_anomaly_tx": matched_anomaly_tx,
            "taxonomy_match_rate": taxonomy_match_rate,
            "dominant_taxonomy": dominant_taxonomy
        }

        for t in TAXONOMY_MAP.keys():
            row[f"{t}_count"] = int(
                anomaly_df[anomaly_df["taxonomy_id"] == t].shape[0]
            )

        results.append(row)

    result_df = pd.DataFrame(results)

    out_path = os.path.join(OUTPUT_DIR, "event_grounded_validation_results.csv")
    result_df.to_csv(out_path, index=False)

    print("[OK] Event-grounded validation saved:", out_path)
    print(result_df)

    return result_df


def load_event_calendar(path):
    events = pd.read_csv(path)

    events["event_datetime"] = pd.to_datetime(
        events["event_date"].astype(str) + " " + events["event_time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    events["expected_taxonomy"] = (
        events["expected_taxonomy"]
        .astype(str)
        .str.replace("&", ",", regex=False)
        .str.replace(" ", "", regex=False)
        .str.split(",")
    )

    events["window_hours"] = pd.to_numeric(
        events["window_hours"],
        errors="coerce"
    ).fillna(12).astype(int)

    return events


def build_doge_price_daily(
        start_date="2021-05-01",
        end_date="2021-05-23",
        output_file="doge_price_daily.csv"
):
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())

    headers = {
        "accept": "application/json"
    }

    url = (
        "https://api.coingecko.com/api/v3/coins/"
        "dogecoin/market_chart/range"
    )

    params = {
        "vs_currency": "usd",
        "from": start_ts,
        "to": end_ts
    }

    r = requests.get(url, params=params,
                     headers=headers)
    r.raise_for_status()

    data = r.json()

    prices = pd.DataFrame(
        data["prices"],
        columns=["timestamp", "price"]
    )

    prices["day"] = (
        pd.to_datetime(prices["timestamp"], unit="ms")
        .dt.date
    )

    doge_daily = (
        prices.groupby("day")["price"]
        .last()
        .reset_index()
    )

    doge_daily.columns = ["day", "close"]

    doge_daily.to_csv(output_file, index=False)

    print(f"[OK] Saved: {output_file}")
    print(doge_daily.head())

    return doge_daily


def main():
    fe_files = sorted(f for f in os.listdir(FEATURE_INPUT_DIR)
                      if f.startswith("fe_chunk_") and f.endswith(".parquet"))

    baseline_models, baseline_scaler, baseline_thresholds = train_benchmark_detectors(fe_files)
    score_benchmark_detectors(fe_files, baseline_models, baseline_scaler, baseline_thresholds)

    baseline_files = sorted(f for f in os.listdir(OUTPUT_DIR)
                            if f.startswith("baseline_scored_chunk_"))
    compare_benchmark_detectors(baseline_files)


if __name__ == "__main__":
    main()