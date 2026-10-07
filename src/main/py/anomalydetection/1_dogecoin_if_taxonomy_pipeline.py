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

import json
import duckdb
import numpy as np
import pandas as pd

from collections import Counter
from neo4j import GraphDatabase, Query
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

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

WORKDIR = "./resources/work_tx_address_gds13_train_centrality"
WORKDIR_READ = WORKDIR
WORKDIR_FEATURE_READ = WORKDIR
os.makedirs(WORKDIR, exist_ok=True)

DATE_FMT = "%Y-%m-%d %H:%M:%S"

IF_FIT_SAMPLE = 500_000
IF_CALIBRATION_SAMPLE = 500_000
IF_N_ESTIMATORS = 150
IF_CONTAMINATION = 0.015
IF_RANDOM_STATE = 42

WRITE_ONLY_ANOMALIES = False

ANOMALY_LABEL_COL = "anomaly_label_gds"
ANOMALY_SCORE_COL = "anomaly_score_gds"

RUN_GDS = False
RUN_EIGENVECTOR = True

GRAPH_COLS = [
    "sender_degree",
    "sender_pageRank",
    "sender_eigenvector",
    "receiver_degree",
    "receiver_pageRank",
    "receiver_eigenvector",
]

MODEL_FEATURE_COLS = [
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
    "T5": "Multi-Profile Overlap",
    "T6": "Uncharacterized Anomaly",
    "NORMAL": "Normal",
}

TRAIN_END = "2021-05-17"
VALID_END = "2021-05-18"
TEST_START = "2021-05-18"

TRAIN_END_ISO = f"{TRAIN_END}T00:00:00"
GDS_GRAPH_NAME = "address_graph_train"
TRAIN_DEGREE_PROPERTY = "degree_train"
TRAIN_PAGERANK_PROPERTY = "pageRank_train"
TRAIN_EIGENVECTOR_PROPERTY = "eigenvector_train"


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


def clear_train_centrality_properties(driver):
    """Remove previously written TRAIN centrality properties before rebuilding the snapshot."""
    print("[INFO] Clearing previous TRAIN-only centrality properties...")

    query = f"""
    CALL apoc.periodic.iterate(
      "MATCH (a:Address)
       WHERE a.{TRAIN_DEGREE_PROPERTY} IS NOT NULL
          OR a.{TRAIN_PAGERANK_PROPERTY} IS NOT NULL
          OR a.{TRAIN_EIGENVECTOR_PROPERTY} IS NOT NULL
       RETURN a",
      "REMOVE a.{TRAIN_DEGREE_PROPERTY},
              a.{TRAIN_PAGERANK_PROPERTY},
              a.{TRAIN_EIGENVECTOR_PROPERTY}",
      {{batchSize: 20000, parallel: false}}
    )
    YIELD batches, total
    RETURN batches, total
    """

    with driver.session() as session:
        rec = session.run(query, timeout=TIMEOUT).single()
        print("[OK] Previous TRAIN centrality properties cleared:", dict(rec))


def drop_gds_graph(driver):
    query = f"""
    CALL gds.graph.drop('{GDS_GRAPH_NAME}', false)
    YIELD graphName
    RETURN graphName
    """

    with driver.session() as session:
        try:
            session.run(query, timeout=TIMEOUT).consume()
            print(f"[OK] old {GDS_GRAPH_NAME} dropped")
        except Exception as e:
            print(f"[INFO] no {GDS_GRAPH_NAME} dropped:", e)


def project_graph(driver):
    """Project all address nodes with training-period relationships only.

    Relationship extraction uses the RawTransaction date index. Query(timeout=0)
    disables the transaction timeout for this long-running projection."""
    print(
        f"[INFO] Creating TRAIN-only Address -> Address GDS projection: "
        f"transaction date < {TRAIN_END} 00:00:00"
    )

    drop_gds_graph(driver)

    train_end_db = f"{TRAIN_END} 00:00:00"

    query = f"""
    CALL gds.graph.project.cypher(
      '{GDS_GRAPH_NAME}',

      '
      MATCH (a:Address)
      RETURN id(a) AS id,
             a.addr_graph_id AS addr_graph_id
      ',

      '
      MATCH (tx:RawTransaction)
      WHERE tx.date < \"{train_end_db}\"
      MATCH (a:Address)-[:SENT]->(tx)-[:RECEIVED]->(b:Address)
      RETURN id(a) AS source,
             id(b) AS target
      '
    )
    YIELD graphName, nodeCount, relationshipCount
    RETURN graphName, nodeCount, relationshipCount
    """

    with driver.session() as session:
        rec = session.run(
            Query(query, timeout=0, metadata={"stage": "train_gds_projection"})
        ).single()
        if rec is None:
            raise RuntimeError("TRAIN-only GDS graph projection returned no result.")
        print("[OK] TRAIN-only GDS graph created:", dict(rec))


def validate_train_graph_projection(driver):
    """Sanity-check TRAIN relationship dates using the indexed RawTransaction.date property."""
    train_end_db = f"{TRAIN_END} 00:00:00"
    query = f"""
    MATCH (tx:RawTransaction)
    WHERE tx.date < $train_end
    RETURN count(tx) AS train_transaction_rows,
           min(tx.date) AS min_train_date,
           max(tx.date) AS max_train_date
    """

    with driver.session() as session:
        rec = session.run(
            Query(query, timeout=0, metadata={"stage": "validate_train_projection"}),
            train_end=train_end_db,
        ).single()
        print("[CHECK] TRAIN graph source rows:", dict(rec))


def compute_degree(driver):
    print("[INFO] Computing TRAIN-only Degree...")

    query = f"""
    CALL gds.degree.write(
      '{GDS_GRAPH_NAME}',
      {{
        writeProperty: '{TRAIN_DEGREE_PROPERTY}',
        orientation: 'UNDIRECTED'
      }}
    )
    YIELD nodePropertiesWritten
    RETURN nodePropertiesWritten
    """

    with driver.session() as session:
        result = session.run(Query(query, timeout=0)).single()
        print("[INFO] TRAIN Degree written:", result["nodePropertiesWritten"])


def compute_pagerank(driver):
    print("[INFO] Computing TRAIN-only PageRank...")

    query = f"""
    CALL gds.pageRank.write(
      '{GDS_GRAPH_NAME}',
      {{
        writeProperty: '{TRAIN_PAGERANK_PROPERTY}',
        maxIterations: 20,
        dampingFactor: 0.85
      }}
    )
    YIELD nodePropertiesWritten, ranIterations
    RETURN nodePropertiesWritten, ranIterations
    """

    with driver.session() as session:
        result = session.run(Query(query, timeout=0)).single()
        print("[INFO] TRAIN PageRank written:", result["nodePropertiesWritten"])
        print("[INFO] Ran iterations:", result["ranIterations"])


def compute_eigenvector(driver):
    print("[INFO] Computing TRAIN-only Eigenvector Centrality...")

    query = f"""
    CALL gds.eigenvector.write(
      '{GDS_GRAPH_NAME}',
      {{
        writeProperty: '{TRAIN_EIGENVECTOR_PROPERTY}',
        maxIterations: 20
      }}
    )
    YIELD nodePropertiesWritten, ranIterations
    RETURN nodePropertiesWritten, ranIterations
    """

    with driver.session() as session:
        result = session.run(Query(query, timeout=0)).single()
        print("[INFO] TRAIN Eigenvector written:", result["nodePropertiesWritten"])
        print("[INFO] Ran iterations:", result["ranIterations"])


def run_gds_centrality(driver):
    """Compute fixed graph-structural features from the TRAIN graph snapshot only."""
    ensure_address_graph_id(driver)
    clear_train_centrality_properties(driver)
    project_graph(driver)
    validate_train_graph_projection(driver)
    compute_degree(driver)
    compute_pagerank(driver)

    if RUN_EIGENVECTOR:
        compute_eigenvector(driver)

    print(
        f"[OK] TRAIN-only centrality completed. "
        f"Properties: {TRAIN_DEGREE_PROPERTY}, {TRAIN_PAGERANK_PROPERTY}, "
        f"{TRAIN_EIGENVECTOR_PROPERTY}"
    )


def new_tx_buffer():
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


def flush_tx_chunk(buf, chunk_idx):
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

    for col in GRAPH_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

    df = df.sort_values(["date", "graph_id"]).reset_index(drop=True)

    out = os.path.join(WORKDIR, f"tx_chunk_{chunk_idx:04d}.parquet")
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


def export_tx_to_parquet(driver):
    print("[INFO] Exporting RawTransaction with fixed TRAIN-only graph centralities...")

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

        coalesce(sender.degree_train, 0.0) AS sender_degree,
        coalesce(sender.pageRank_train, 0.0) AS sender_pageRank,
        coalesce(sender.eigenvector_train, 0.0) AS sender_eigenvector,

        coalesce(receiver.degree_train, 0.0) AS receiver_degree,
        coalesce(receiver.pageRank_train, 0.0) AS receiver_pageRank,
        coalesce(receiver.eigenvector_train, 0.0) AS receiver_eigenvector
    """

    chunk_idx = 0
    buf = new_tx_buffer()

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
                    flush_tx_chunk(buf, chunk_idx)
                    print(f"[OK] wrote chunk {chunk_idx}")
                    chunk_idx += 1
                    buf = new_tx_buffer()

            print(f"[INFO] Range completed: rows={row_count}")

            start_id = end_id

        if buf["graph_id"]:
            flush_tx_chunk(buf, chunk_idx)
            print(f"[OK] wrote final chunk {chunk_idx}")

    print("[OK] Export completed")


def sort_tx_parquets_by_date_graphid():
    print("[INFO] Sorting tx_chunk parquet files by date, graph_id using DuckDB...")

    input_pattern = os.path.join(WORKDIR, "tx_chunk_*.parquet")
    sorted_dir = os.path.join(WORKDIR, "sorted_chunks")
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
        *GRAPH_COLS,
    ]

    for c in numeric_cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

    for c in GRAPH_COLS:
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


def feature_engineer_parquets():
    print("[INFO] Feature engineering parquet chunks with market-aligned global buckets...")

    sorted_dir = os.path.join(WORKDIR_READ, "sorted_chunks")

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

    for col in GRAPH_COLS:
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
        f for f in os.listdir(WORKDIR)
        if f.startswith("fe_chunk_") and f.endswith(".parquet")
    ]

    for f in old_fe:
        os.remove(os.path.join(WORKDIR, f))

    chunk_size = 250_000

    for fe_idx, start in enumerate(range(0, len(df_all), chunk_size)):
        chunk = df_all.iloc[start:start + chunk_size].copy()
        out = os.path.join(WORKDIR, f"fe_chunk_{fe_idx:04d}.parquet")
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
            os.path.join(WORKDIR_FEATURE_READ, fn),
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

        "same_timestamp_count_q50": float(big["same_timestamp_count"].quantile(0.50)),

    }

    print("[OK] global thresholds:", thresholds)

    with open(os.path.join(WORKDIR, "global_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    del big, parts
    return thresholds


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


def _load_temporal_stratified_train_samples(
        fe_files,
        fit_rows,
        calibration_rows,
        random_state,
):
    """Create disjoint training-only fit and calibration samples, stratified by day."""
    print(
        f"[INFO] Building temporal-stratified TRAIN samples: "
        f"fit={fit_rows}, calibration={calibration_rows}"
    )

    train_end = pd.Timestamp(TRAIN_END)

    # Pass 1: count available TRAIN rows by day.
    day_counts = {}
    for fn in fe_files:
        df_date = pd.read_parquet(
            os.path.join(WORKDIR_FEATURE_READ, fn),
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
    cols = ["date", *MODEL_FEATURE_COLS]

    # Pass 2: bounded random-key reservoir per day.
    for fn in fe_files:
        df = pd.read_parquet(
            os.path.join(WORKDIR_FEATURE_READ, fn),
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
            candidate = g[MODEL_FEATURE_COLS].copy()
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
            random_state=random_state + i + 1,
        ).reset_index(drop=True)

        fit_day = r.iloc[:q_fit][MODEL_FEATURE_COLS]
        cal_day = r.iloc[q_fit:q_fit + q_cal][MODEL_FEATURE_COLS]

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
        os.path.join(WORKDIR, "if_sampling_by_day.csv"),
        index=False,
    )

    print("[OK] Temporal-stratified TRAIN sampling completed.")
    print(diag.to_string(index=False))
    print(f"[OK] fit_rows={len(X_fit)}, calibration_rows={len(X_cal)}")

    return X_fit, X_cal


def train_isolation_forest(fe_files):
    print("[INFO] Training IsolationForest with independent TRAIN calibration...")

    X_fit, X_cal = _load_temporal_stratified_train_samples(
        fe_files=fe_files,
        fit_rows=IF_FIT_SAMPLE,
        calibration_rows=IF_CALIBRATION_SAMPLE,
        random_state=IF_RANDOM_STATE,
    )

    scaler = RobustScaler()
    X_fit_scaled = scaler.fit_transform(X_fit).astype(np.float32)
    X_cal_scaled = scaler.transform(X_cal).astype(np.float32)

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_fit_scaled)

    calibration_scores = -model.decision_function(X_cal_scaled)
    if_threshold = float(np.percentile(
        calibration_scores,
        100.0 * (1.0 - IF_CONTAMINATION),
    ))

    calibration_anomalies = int((calibration_scores > if_threshold).sum())
    calibration_rate = 100.0 * calibration_anomalies / len(calibration_scores)

    print(
        f"[OK] Main IF trained. fit_rows={len(X_fit_scaled)} "
        f"calibration_rows={len(X_cal_scaled)} "
        f"threshold={if_threshold:.9f} "
        f"calibration_rate={calibration_rate:.3f}%"
    )

    gt_rate = 100.0 * np.count_nonzero(calibration_scores > if_threshold) / len(calibration_scores)
    eq_rate = 100.0 * np.count_nonzero(calibration_scores == if_threshold) / len(calibration_scores)

    print(
        "[IF CAL DIAG]",
        f"min={calibration_scores.min():.12f}",
        f"p50={np.percentile(calibration_scores, 50):.12f}",
        f"p95={np.percentile(calibration_scores, 95):.12f}",
        f"p98.5={np.percentile(calibration_scores, 98.5):.12f}",
        f"p99={np.percentile(calibration_scores, 99):.12f}",
        f"max={calibration_scores.max():.12f}",
        f"gt={gt_rate:.4f}%",
        f"eq={eq_rate:.4f}%"
    )

    if HAS_JOBLIB:
        joblib.dump(model, os.path.join(WORKDIR, "if_model.joblib"))
        joblib.dump(scaler, os.path.join(WORKDIR, "if_scaler.joblib"))

    pd.DataFrame([{
        "fit_rows": len(X_fit_scaled),
        "calibration_rows": len(X_cal_scaled),
        "threshold": if_threshold,
        "contamination": IF_CONTAMINATION,
        "calibration_anomalies": calibration_anomalies,
        "calibration_anomaly_rate_pct": calibration_rate,
        "threshold_source": "independent_temporal_stratified_TRAIN_calibration",
        "valid_test_used_for_threshold": False,
    }]).to_csv(
        os.path.join(WORKDIR, "if_train_calibration.csv"),
        index=False,
    )

    return model, scaler, if_threshold


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
    """Score all periods using the fixed threshold learned from TRAIN calibration."""
    print("[INFO] Scoring IsolationForest with fixed TRAIN-only threshold...")
    print(f"[INFO] IF TRAIN-calibration threshold={if_threshold:.15f}")

    out_idx = 0

    for fn in fe_files:
        path = os.path.join(WORKDIR_FEATURE_READ, fn)
        df = pd.read_parquet(path, engine="fastparquet")
        df = add_model_features(df)

        X = (
            df[MODEL_FEATURE_COLS]
            .astype(np.float32)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        Xs = scaler.transform(X).astype(np.float32)
        score = -model.decision_function(Xs)

        df[ANOMALY_SCORE_COL] = score.astype(np.float32)
        # Strict upper-tail rule: only scores strictly above the calibrated percentile.
        df[ANOMALY_LABEL_COL] = (score > if_threshold).astype(np.int8)

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

        # IMPORTANT: compare LOG features with LOG thresholds.
        market_volatility_flag = (
                (df["value_std_15m_log"] >= thresholds["value_std_15m_log_q95"]) |
                (df["active_sender_15m_log"] >= thresholds["active_sender_15m_log_q95"])
        )

        rapid_sender_flag = (
                (df["seconds_since_last"] <= thresholds["sec_q10"]) &
                (df["active_sender_15m_log"] >= thresholds["active_sender_15m_log_q95"]) &
                (df["tx_count_15min_z"] >= thresholds["z15_q90"])
        )

        df["time_anomaly_flag"] = (
                (market_spike_flag & market_volatility_flag)
                |
                (market_density_flag & rapid_sender_flag & (df[ANOMALY_LABEL_COL] == 1))
        ).astype(np.int8)

        high_graph = (
                (df["max_degree_log"] >= thresholds["max_degree_log_q995"])
                |
                (
                        (df["max_pageRank_log"] >= thresholds["max_pageRank_log_q99"])
                        &
                        (df["max_eigenvector_log"] >= thresholds["max_eigenvector_log_q99"])
                )
        )

        df["graph_anomaly_flag"] = (
                high_graph
                &
                (
                        (df[ANOMALY_LABEL_COL] == 1)
                        |
                        (df["time_anomaly_flag"] == 1)
                )
        ).astype(np.int8)

        # Keep this explicit: IF detects anomalies; temporal/graph flags characterize them.
        df["final_anomaly"] = df[ANOMALY_LABEL_COL].astype(np.int8)

        df["anomaly_label_market"] = (
                market_volatility_flag | market_spike_flag
        ).astype(np.int8)

        out = os.path.join(WORKDIR, f"base_scored_chunk_{out_idx:04d}.parquet")
        df.to_parquet(out, engine="fastparquet", index=False)

        for split_name in ["TRAIN", "VALID", "TEST"]:
            sdf = df[df["dataset_split"] == split_name]
            if len(sdf) == 0:
                continue

            print(
                f"[{split_name}] "
                f"rows={len(sdf)} "
                f"if={int(sdf[ANOMALY_LABEL_COL].sum())} "
                f"time={int(sdf['time_anomaly_flag'].sum())} "
                f"graph={int(sdf['graph_anomaly_flag'].sum())} "
                f"final={int(sdf['final_anomaly'].sum())}"
            )

        out_idx += 1

    return float(if_threshold)


PROFILE_RULE_VERSION = "behavior_profiles_v1"
PROFILE_QUANTILES = {"stress_lenient": .90, "local_lenient": .93,
                     "baseline": .95, "local_strict": .97,
                     "stress_strict": .995}
PROFILE_SAMPLE_SIZE = 200_000
PROFILE_IDS = ("T1", "T2", "T3", "T4", "T5", "T6")
PROFILE_FEATURES = [
    "value", "tx_count_15min_z", "active_sender_15m_log",
    *[f"{side}_{metric}_log" for side in ("sender", "receiver")
      for metric in ("degree", "pageRank", "eigenvector")],
]


def fit_behavior_profile_thresholds():
    """Calibrate all cutoffs on TRAIN only.

    T1/T3 use one observation per occupied 15-minute window. T2 and T4 use a
    uniform TRAIN transaction sample. Centrality thresholds are therefore
    transaction-endpoint weighted; the exact weighting is saved with results.
    """
    files = sorted(f for f in os.listdir(WORKDIR_FEATURE_READ)
                   if f.startswith("fe_chunk_") and f.endswith(".parquet"))
    if not files:
        raise RuntimeError("TRAIN feature parquet files are required.")
    rng = np.random.default_rng(42)
    sample = pd.DataFrame()
    windows = {}
    train_rows = 0
    for fn in files:
        df = pd.read_parquet(os.path.join(WORKDIR_FEATURE_READ, fn),
                             columns=["date", *PROFILE_FEATURES],
                             engine="fastparquet")
        date = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.loc[date < pd.Timestamp(TRAIN_END, tz="UTC")].copy()
        if df.empty:
            continue
        for col in PROFILE_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(df[PROFILE_FEATURES].to_numpy(dtype=float)).all():
            raise ValueError(f"Missing/non-finite profile features in {fn}")
        if (df["value"] < 0).any():
            raise ValueError(f"Negative transformed transfer value in {fn}")
        df["_window"] = date.loc[df.index].dt.floor("15min")
        for key, group in df.groupby("_window", sort=False):
            values = group[["tx_count_15min_z", "active_sender_15m_log"]].drop_duplicates()
            if len(values) != 1:
                raise ValueError(f"Window features are not constant in {key}")
            pair = tuple(float(v) for v in values.iloc[0])
            if key in windows and not np.allclose(windows[key], pair):
                raise ValueError(f"Conflicting values for window {key}")
            windows[key] = pair
        train_rows += len(df)
        tx = df[["value", *PROFILE_FEATURES[3:]]].copy()
        tx["_priority"] = rng.random(len(tx))
        sample = pd.concat([sample, tx], ignore_index=True).nsmallest(
            PROFILE_SAMPLE_SIZE, "_priority")
    if not windows or sample.empty:
        raise RuntimeError("No eligible TRAIN observations for taxonomy thresholds.")
    win = pd.DataFrame(windows.values(), columns=["density", "senders"])
    series = {"density": win["density"], "senders": win["senders"],
              "value": sample["value"]}
    for metric in ("degree", "pageRank", "eigenvector"):
        series[metric] = sample[[f"sender_{metric}_log",
                                 f"receiver_{metric}_log"]].stack()
    scenarios = {
        name: {key: float(values.quantile(q)) for key, values in series.items()}
        for name, q in PROFILE_QUANTILES.items()
    }
    result = {
        "rule_version": PROFILE_RULE_VERSION,
        "train_end_exclusive": TRAIN_END,
        "train_rows": train_rows,
        "train_windows": len(windows),
        "transaction_sample_rows": len(sample),
        "random_state": 42,
        "baseline_quantile": .95,
        "window_weighting": "one observation per occupied 15-minute TRAIN window",
        "value_weighting": "uniform TRAIN transaction sample",
        "centrality_weighting": "sender and receiver endpoints from a uniform TRAIN transaction sample",
        "value_scale": "stored value column (log1p DOGE in this pipeline)",
        "scenarios": scenarios,
        "thresholds": scenarios["baseline"],
    }
    path = os.path.join(WORKDIR, "profile_thresholds.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"[OK] Saved TRAIN-only profile thresholds: {path}")
    return result


def load_behavior_profile_thresholds():
    path = os.path.join(WORKDIR, "profile_thresholds.json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("rule_version") != PROFILE_RULE_VERSION:
        raise ValueError("Taxonomy rule-version mismatch")
    if data.get("train_end_exclusive") != TRAIN_END:
        raise ValueError("Taxonomy TRAIN boundary mismatch")
    return data


def behavior_profile_flags(df, thresholds):
    missing = sorted(set(PROFILE_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required profile features: {missing}")
    x = df[PROFILE_FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite profile features; rows are not silently assigned to T6")
    flags = pd.DataFrame(index=df.index)
    flags["T1"] = ((x["tx_count_15min_z"] >= thresholds["density"])
                   & (x["tx_count_15min_z"] > 0))
    flags["T2"] = ((x["value"] >= thresholds["value"])
                   & (x["value"] > 0))
    flags["T3"] = ((x["active_sender_15m_log"] >= thresholds["senders"])
                   & (x["active_sender_15m_log"] > 0))
    endpoint_high = []
    for side in ("sender", "receiver"):
        votes = sum(
            ((x[f"{side}_{metric}_log"] >= thresholds[metric])
             & (x[f"{side}_{metric}_log"] > 0)).astype("int8")
            for metric in ("degree", "pageRank", "eigenvector")
        )
        endpoint_high.append(votes >= 2)
    flags["T4"] = endpoint_high[0] | endpoint_high[1]
    return flags


def classify_behavior_profiles(df, anomaly_col, thresholds):
    out = df.copy()
    out["taxonomy_id"] = "NORMAL"
    out["taxonomy_components"] = ""
    out["taxonomy_n_components"] = np.int8(0)
    out["taxonomy_rule_version"] = PROFILE_RULE_VERSION
    for tid in ("T1", "T2", "T3", "T4"):
        out[f"{tid}_match"] = False
    anomaly = pd.to_numeric(out[anomaly_col], errors="raise").eq(1)
    if anomaly.any():
        flags = behavior_profile_flags(out.loc[anomaly], thresholds)
        n = flags.sum(axis=1)
        labels = pd.Series("T6", index=flags.index, dtype="object")
        labels.loc[n >= 2] = "T5"
        for tid in flags:
            labels.loc[(n == 1) & flags[tid]] = tid
            out.loc[anomaly, f"{tid}_match"] = flags[tid].to_numpy()
        bits = sum(flags[tid].astype("int8") * (1 << i)
                   for i, tid in enumerate(flags))
        combinations = {
            value: ",".join(tid for i, tid in enumerate(flags)
                            if value & (1 << i))
            for value in range(16)
        }
        out.loc[anomaly, "taxonomy_id"] = labels.to_numpy()
        out.loc[anomaly, "taxonomy_components"] = bits.map(combinations).to_numpy()
        out.loc[anomaly, "taxonomy_n_components"] = n.to_numpy(dtype=np.int8)
    out["anomaly_type"] = out["taxonomy_id"].map(TAXONOMY_MAP)
    out["taxonomy_reason"] = np.where(
        anomaly,
        out["anomaly_type"] + "; components=" + out["taxonomy_components"],
        "-",
    )
    return out


def apply_behavior_taxonomy(base_files):
    thresholds = load_behavior_profile_thresholds()["thresholds"]
    totals = {tid: 0 for tid in PROFILE_IDS}
    daily_parts, combination_parts = [], []
    for idx, fn in enumerate(base_files):
        df = pd.read_parquet(os.path.join(WORKDIR, fn), engine="fastparquet")
        df = classify_behavior_profiles(df, "final_anomaly", thresholds)
        out = os.path.join(WORKDIR, f"scored_chunk_{idx:04d}.parquet")
        df.to_parquet(out, engine="fastparquet", index=False)
        anomalies = df.loc[pd.to_numeric(df["final_anomaly"]).eq(1)].copy()
        counts = anomalies["taxonomy_id"].value_counts()
        for tid in totals:
            totals[tid] += int(counts.get(tid, 0))
        anomalies["day"] = pd.to_datetime(anomalies["date"], utc=True).dt.strftime("%Y-%m-%d")
        daily_parts.append(anomalies.groupby(["day", "taxonomy_id"]).size().rename("count").reset_index())
        combination_parts.append(anomalies.groupby(
            ["taxonomy_id", "taxonomy_components"]).size().rename("count").reset_index())
        print(f"[OK] Taxonomy chunk {idx}: anomalies={len(anomalies)}")
    summary = pd.DataFrame([
        {"taxonomy_id": tid, "anomaly_type": TAXONOMY_MAP[tid], "count": count}
        for tid, count in totals.items()
    ])
    total = int(summary["count"].sum())
    summary["percentage"] = summary["count"] * 100.0 / total if total else 0.0
    summary.to_csv(os.path.join(WORKDIR, "final_taxonomy_summary.csv"), index=False)
    if daily_parts:
        daily = pd.concat(daily_parts).groupby(
            ["day", "taxonomy_id"])["count"].sum().unstack(fill_value=0)
        daily = daily.reindex(columns=list(PROFILE_IDS), fill_value=0)
        daily["total"] = daily.sum(axis=1)
        daily.to_csv(os.path.join(WORKDIR, "daily_taxonomy_counts.csv"))
        pct = daily[list(PROFILE_IDS)].div(
            daily["total"].replace(0, np.nan), axis=0).mul(100).fillna(0)
        pct.to_csv(os.path.join(WORKDIR, "daily_taxonomy_percentages.csv"))
        combos = pd.concat(combination_parts).groupby(
            ["taxonomy_id", "taxonomy_components"], as_index=False)["count"].sum()
        combos.to_csv(os.path.join(WORKDIR, "taxonomy_component_combinations.csv"), index=False)
    print(summary.to_string(index=False))
    return summary


def rebuild_behavior_profiles():
    base_files = sorted(f for f in os.listdir(WORKDIR)
                        if f.startswith("base_scored_chunk_") and f.endswith(".parquet"))
    if not base_files:
        raise RuntimeError("Run the IF scoring stage first; base_scored_chunk files are missing")
    fit_behavior_profile_thresholds()
    return apply_behavior_taxonomy(base_files)


def rebuild_features_with_train_centrality():
    """Recompute training-period centralities, export all records, sort them,
    and rebuild features. Validation and test records use fixed centralities."""
    print("\n[INFO] Rebuilding features with TRAIN-only graph centrality...\n")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        ensure_indexes(driver)
        run_gds_centrality(driver)
        export_tx_to_parquet(driver)
    finally:
        driver.close()

    sort_tx_parquets_by_date_graphid()
    feature_engineer_parquets()

    print(
        f"\n[OK] Fresh leakage-controlled features created in: {WORKDIR}\n"
    )


def train_if_and_profile_taxonomy():
    fe_files = sorted(f for f in os.listdir(WORKDIR_FEATURE_READ)
                      if f.startswith("fe_chunk_") and f.endswith(".parquet"))
    if not fe_files:
        raise RuntimeError("No fe_chunk parquet files found")
    detector_thresholds = compute_global_thresholds(fe_files)
    model, scaler, if_threshold = train_isolation_forest(fe_files)
    score_isolation_forest(fe_files, model, scaler, detector_thresholds, if_threshold)
    fit_behavior_profile_thresholds()
    base_files = sorted(f for f in os.listdir(WORKDIR)
                        if f.startswith("base_scored_chunk_") and f.endswith(".parquet"))
    return apply_behavior_taxonomy(base_files)


def print_daily_anomaly_summary():
    print("\n[INFO] Daily anomaly summary...\n")

    scored_files = sorted([
        f for f in os.listdir(WORKDIR)
        if f.startswith("base_scored_chunk_") and f.endswith(".parquet")
    ])

    if not scored_files:
        raise RuntimeError("No scored_chunk parquet files found.")

    parts = []

    for fn in scored_files:
        df = pd.read_parquet(
            os.path.join(WORKDIR, fn),
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

    out_path = os.path.join(WORKDIR, "daily_anomaly_summary.csv")
    daily_summary.to_csv(out_path, index=False)

    print(f"\n[OK] Saved daily anomaly summary to: {out_path}")


def event_comparison_validation():
    print("[INFO] Event Comparison Validation")

    scored_files = sorted([
        f for f in os.listdir(WORKDIR)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    ])

    configured_event_file = os.path.join(WORKDIR_FEATURE_READ, f"event_calendar.csv")
    if os.path.isabs(configured_event_file):
        event_calendar_path = configured_event_file
    else:
        event_calendar_path = configured_event_file
    if not os.path.exists(event_calendar_path):
        raise FileNotFoundError(
            f"Event calendar not found: {event_calendar_path}. "
            "Set EVENT_CALENDAR_FILE to an absolute path or a filename "
            "under WORKDIR_FEATURE_READ."
        )

    event_results = event_grounded_validation(
        scored_files=scored_files,
        event_calendar_path=os.path.join(WORKDIR_FEATURE_READ, f"event_calendar.csv")
    )


def write_back(driver):
    print("[INFO] Writing results back to Neo4j...")

    scored_files = sorted([
        f for f in os.listdir(WORKDIR)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    ])

    if not scored_files:
        raise RuntimeError("No scored_chunk parquet files found.")

    update_query = f"""
    UNWIND $rows AS row
    MATCH (t:RawTransaction {{graph_id: row.graph_id}})
    SET
        t.{ANOMALY_LABEL_COL} = row.label,
        t.{ANOMALY_SCORE_COL} = row.score,
        t.final_label = row.final_label,
        t.time_anomaly_flag = row.time_flag,
        t.graph_anomaly_flag = row.graph_flag,

        t.seconds_since_last = row.ssl,
        t.tx_count_15min = row.c15,
        t.tx_count_60min = row.c60,
        t.tx_count_15min_delta = row.d15,
        t.tx_count_15min_z = row.z15,

        t.sender_degree = row.sender_degree,
        t.sender_pageRank = row.sender_pageRank,
        t.sender_eigenvector = row.sender_eigenvector,

        t.receiver_degree = row.receiver_degree,
        t.receiver_pageRank = row.receiver_pageRank,
        t.receiver_eigenvector = row.receiver_eigenvector,


        t.taxonomy_id = row.taxonomy_id,
        t.anomaly_type = row.anomaly_type,
        t.taxonomy_reason = row.taxonomy_reason,
        t.taxonomy_components = row.taxonomy_components,
        t.taxonomy_n_components = row.taxonomy_n_components,
        t.taxonomy_rule_version = row.taxonomy_rule_version,

        t.value_std_15m_log = row.value_std_15m_log,
        t.active_sender_15m_log = row.active_sender_15m_log,
        t.anomaly_label_market = row.market_label
    """

    cols = [
        "graph_id",
        "seconds_since_last",
        "tx_count_15min",
        "tx_count_60min",
        "tx_count_15min_delta",
        "tx_count_15min_z",
        "sender_degree",
        "sender_pageRank",
        "sender_eigenvector",
        "receiver_degree",
        "receiver_pageRank",
        "receiver_eigenvector",
        ANOMALY_LABEL_COL,
        ANOMALY_SCORE_COL,
        "final_anomaly",
        "time_anomaly_flag",
        "graph_anomaly_flag",
        "taxonomy_id",
        "anomaly_type",
        "taxonomy_reason",
        "taxonomy_components",
        "taxonomy_n_components",
        "taxonomy_rule_version",
        "value_std_15m_log",
        "active_sender_15m_log",
        "anomaly_label_market",
    ]

    with driver.session() as session:
        for fn in scored_files:
            df = pd.read_parquet(
                os.path.join(WORKDIR, fn),
                columns=cols,
                engine="fastparquet",
            )

            if WRITE_ONLY_ANOMALIES:
                df = df[df["final_anomaly"] == 1]

            n = len(df)

            if n == 0:
                print(f"[INFO] skipped {fn}, no anomaly")
                continue

            for start in range(0, n, WRITE_BATCH):
                part = df.iloc[start:start + WRITE_BATCH]

                rows = [
                    {
                        "graph_id": int(row.graph_id),
                        "ssl": float(row.seconds_since_last),
                        "c15": float(row.tx_count_15min),
                        "c60": float(row.tx_count_60min),
                        "d15": float(row.tx_count_15min_delta),
                        "z15": float(row.tx_count_15min_z),

                        "sender_degree": float(row.sender_degree),
                        "sender_pageRank": float(row.sender_pageRank),
                        "sender_eigenvector": float(row.sender_eigenvector),

                        "receiver_degree": float(row.receiver_degree),
                        "receiver_pageRank": float(row.receiver_pageRank),
                        "receiver_eigenvector": float(row.receiver_eigenvector),

                        "label": int(getattr(row, ANOMALY_LABEL_COL)),
                        "score": float(getattr(row, ANOMALY_SCORE_COL)),
                        "final_label": int(row.final_anomaly),
                        "time_flag": int(row.time_anomaly_flag),
                        "graph_flag": int(row.graph_anomaly_flag),

                        "taxonomy_id": str(row.taxonomy_id),
                        "anomaly_type": str(row.anomaly_type),
                        "taxonomy_reason": str(row.taxonomy_reason),
                        "taxonomy_components": str(row.taxonomy_components),
                        "taxonomy_n_components": int(row.taxonomy_n_components),
                        "taxonomy_rule_version": str(row.taxonomy_rule_version),
                        "value_std_15m_log": float(row.value_std_15m_log),
                        "active_sender_15m_log": float(row.active_sender_15m_log),
                        "market_label": int(row.anomaly_label_market),
                    }
                    for row in part.itertuples(index=False)
                ]

                session.run(update_query, rows=rows, timeout=TIMEOUT).consume()

            print(f"[INFO] wrote back {fn} rows={n}")

    print("[OK] Write-back completed")


def print_daily_taxonomy_distribution(
        input_pattern="scored_chunk_*.csv",
        count_output_file="daily_taxonomy_counts.csv",
        percentage_output_file="daily_taxonomy_percentages.csv"
):
    files = sorted([
        f for f in os.listdir(WORKDIR)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    ])

    if not files:
        print(f"[WARN] No files found: {input_pattern}")
        return pd.DataFrame(), pd.DataFrame()

    dfs = []

    for f in files:
        try:
            df = pd.read_parquet(
                os.path.join(WORKDIR, f),
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

    for tax in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        if tax not in count_pivot.columns:
            count_pivot[tax] = 0

    count_pivot["total"] = (
            count_pivot["T1"] +
            count_pivot["T2"] +
            count_pivot["T3"] +
            count_pivot["T4"] +
            count_pivot["T5"] +
            count_pivot["T6"]
    )

    count_pivot = count_pivot[
        ["day", "T1", "T2", "T3", "T4", "T5", "T6", "total"]
    ].sort_values("day")

    percentage_pivot = count_pivot.copy()

    for tax in ["T1", "T2", "T3", "T4", "T5", "T6"]:
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
        f for f in os.listdir(WORKDIR)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    ])

    if not scored_files:
        raise RuntimeError("No scored_chunk parquet files found.")

    rows = []

    for fn in scored_files:
        df = pd.read_parquet(
            os.path.join(WORKDIR, fn),
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

    wanted = ["T1", "T2", "T3", "T4", "T5", "T6"]

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

    out_path = os.path.join(WORKDIR, "final_taxonomy_summary.csv")
    summary.to_csv(out_path, index=False)

    print(f"[OK] Saved taxonomy summary to: {out_path}")


def print_validation_queries():
    print("\nValidation Cypher queries:\n")

    print(f"""
MATCH (t:RawTransaction)
RETURN
  count(*) AS total,
  count(t.{ANOMALY_LABEL_COL}) AS if_labeled,
  sum(t.{ANOMALY_LABEL_COL}) AS if_anomaly,
  sum(t.time_anomaly_flag) AS time_anomaly,
  sum(t.graph_anomaly_flag) AS graph_anomaly,
  sum(t.final_label) AS final_anomaly;
""")

    print("""
MATCH (t:RawTransaction)
WHERE t.final_label = 1
RETURN
  t.taxonomy_id AS taxonomy_id,
  t.anomaly_type AS anomaly_type,
  t.taxonomy_components AS taxonomy_components,
  count(*) AS cnt
ORDER BY cnt DESC;
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

    # Establish the actual observation period once. Event windows that are not
    # fully observed are excluded instead of being reported as zero matches.
    file_bounds = []
    for fn in scored_files:
        dates = pd.read_parquet(
            os.path.join(WORKDIR, fn),
            columns=["date"],
        )["date"]
        dates = pd.to_datetime(dates, errors="coerce", utc=True).dt.tz_convert(None)
        dates = dates.dropna()
        if not dates.empty:
            file_bounds.append((fn, dates.min(), dates.max()))

    if not file_bounds:
        raise RuntimeError("No valid transaction dates were found in scored parquet files.")

    data_start = min(x[1] for x in file_bounds)
    # Treat the last stored instant as observed by adding one nanosecond to the
    # half-open study interval [data_start, data_end).
    data_end = max(x[2] for x in file_bounds) + pd.Timedelta(nanoseconds=1)
    excluded_rows = []

    for _, event in events.iterrows():
        event_id = event["event_id"]
        event_name = event["event_name"]
        event_type = event["event_type"]
        event_time = pd.Timestamp(event["event_datetime"])
        window_hours = float(event["window_hours"])
        time_precision = str(event["time_precision"])
        expected = tuple(event["expected_taxonomy"])

        if time_precision == "day":
            start_time = event_time.normalize()
            end_time = start_time + pd.Timedelta(days=1)
        else:
            delta = pd.Timedelta(hours=window_hours)
            start_time = event_time - delta
            end_time = event_time + delta

        effective_window_hours = (end_time - start_time).total_seconds() / 3600.0

        if start_time < data_start or end_time > data_end:
            excluded_rows.append({
                "event_id": event_id,
                "event_name": event_name,
                "event_type": event_type,
                "event_datetime": event_time,
                "time_precision": time_precision,
                "window_hours": window_hours,
                "effective_window_hours": effective_window_hours,
                "expected_taxonomy": ",".join(expected),
                "window_start": start_time,
                "window_end": end_time,
                "data_start": data_start,
                "data_end": data_end,
                "exclusion_reason": "complete event window falls outside the observed transaction period",
            })
            continue

        event_rows = []

        for fn, chunk_start, chunk_end in file_bounds:
            if chunk_end < start_time or chunk_start >= end_time:
                continue

            df = pd.read_parquet(os.path.join(WORKDIR, fn))

            df["date"] = pd.to_datetime(
                df["date"], errors="coerce", utc=True
            ).dt.tz_convert(None)

            sub = df[
                (df["date"] >= start_time) &
                (df["date"] < end_time)
                ]

            if len(sub) > 0:
                event_rows.append(sub)

        if not event_rows:
            results.append({
                "event_id": event_id,
                "event_name": event_name,
                "event_type": event_type,
                "event_datetime": event_time,
                "time_precision": time_precision,
                "window_hours": window_hours,
                "effective_window_hours": effective_window_hours,
                "expected_taxonomy": ",".join(expected),
                "window_start": start_time,
                "window_end": end_time,
                "eligibility_status": "eligible_no_transactions",
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
            (event_df[ANOMALY_LABEL_COL] == 1)
            ].copy()

        anomaly_tx = len(anomaly_df)

        if anomaly_tx > 0:
            taxonomy_counts = anomaly_df["taxonomy_id"].value_counts()
            dominant_taxonomy = taxonomy_counts.index[0]

            component_flags = pd.DataFrame(False, index=anomaly_df.index,
                                           columns=["T1", "T2", "T3", "T4"])
            for tid in component_flags.columns:
                flag_col = f"{tid}_match"
                if flag_col in anomaly_df.columns:
                    component_flags[tid] = (
                        anomaly_df[flag_col].fillna(False).astype(bool)
                    )
                elif "taxonomy_components" in anomaly_df.columns:
                    component_flags[tid] = anomaly_df["taxonomy_components"].fillna("").map(
                        lambda value, target=tid: target in {
                            x.strip() for x in str(value).split(",") if x.strip()
                        }
                    )
                else:
                    component_flags[tid] = anomaly_df["taxonomy_id"].eq(tid)

            matched_mask = component_flags[list(expected)].any(axis=1)
            matched_anomaly_tx = int(matched_mask.sum())
            component_counts = component_flags.sum(axis=0).astype(int)
            dominant_component = str(component_counts.idxmax())

            taxonomy_match_rate = round(
                100.0 * matched_anomaly_tx / anomaly_tx, 3
            )
        else:
            dominant_taxonomy = None
            dominant_component = None
            component_counts = pd.Series(0, index=["T1", "T2", "T3", "T4"])
            matched_anomaly_tx = 0
            taxonomy_match_rate = 0

        anomaly_rate = round(100.0 * anomaly_tx / total_tx, 3)

        row = {
            "event_id": event_id,
            "event_name": event_name,
            "event_type": event_type,
            "event_datetime": event_time,
            "time_precision": time_precision,
            "window_hours": window_hours,
            "effective_window_hours": effective_window_hours,
            "expected_taxonomy": ",".join(expected),
            "window_start": start_time,
            "window_end": end_time,
            "eligibility_status": "eligible",
            "total_tx": total_tx,
            "anomaly_tx": anomaly_tx,
            "anomaly_rate": anomaly_rate,
            "matched_anomaly_tx": matched_anomaly_tx,
            "taxonomy_match_rate": taxonomy_match_rate,
            "dominant_taxonomy": dominant_taxonomy,
            "dominant_component": dominant_component,
            "match_definition": "any expected atomic T1-T4 component",
        }

        for t in TAXONOMY_MAP.keys():
            row[f"{t}_count"] = int(
                anomaly_df[anomaly_df["taxonomy_id"] == t].shape[0]
            )

        for t in ("T1", "T2", "T3", "T4"):
            row[f"{t}_component_count"] = int(component_counts[t])

        results.append(row)

    result_df = pd.DataFrame(results)

    out_path = os.path.join(WORKDIR, "event_grounded_validation_results.csv")
    result_df.to_csv(out_path, index=False)

    excluded_path = os.path.join(
        WORKDIR, "event_grounded_validation_excluded_events.csv"
    )
    pd.DataFrame(excluded_rows).to_csv(excluded_path, index=False)

    print("[OK] Event-grounded validation saved:", out_path)
    print("[OK] Excluded event windows saved:", excluded_path)
    print(result_df)

    return result_df


def load_event_calendar(path):
    events = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "event_id", "event_date", "event_time", "event_name", "event_type",
        "expected_taxonomy", "window_hours",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event calendar is missing columns: {sorted(missing)}")

    event_date = pd.to_datetime(
        events["event_date"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    time_text = events["event_time"].astype("string").str.strip()
    legacy_day_token = time_text.str.lower().isin(
        {"time_precision=day", "day", "all_day", "all-day", "daily"}
    )
    time_missing = time_text.isna() | time_text.eq("") | legacy_day_token
    if "time_precision" in events.columns:
        declared_precision = events["time_precision"].astype("string").str.strip().str.lower()
        declared_day = declared_precision.isin(
            {"day", "date", "all_day", "all-day", "daily"}
        )
    else:
        declared_day = pd.Series(False, index=events.index)

    events["time_precision"] = np.where(time_missing | declared_day, "day", "time")
    events["event_datetime"] = pd.NaT
    timed = events["time_precision"].eq("time")
    timed_text = (
            events.loc[timed, "event_date"].astype(str).str.strip()
            + " " + time_text.loc[timed]
    )
    events.loc[timed, "event_datetime"] = pd.to_datetime(
        timed_text, dayfirst=True, errors="coerce", utc=True
    ).dt.tz_convert(None)
    events.loc[~timed, "event_datetime"] = event_date.loc[~timed]

    events["expected_taxonomy"] = (
        events["expected_taxonomy"]
        .astype(str)
        .str.replace("&", ",", regex=False)
        .str.replace(" ", "", regex=False)
        .str.split(",")
        .map(lambda values: tuple(dict.fromkeys(
            value for value in values if value in {"T1", "T2", "T3", "T4"}
        )))
    )

    events["window_hours"] = pd.to_numeric(
        events["window_hours"],
        errors="coerce"
    )

    invalid = (
            events["event_datetime"].isna()
            | events["expected_taxonomy"].map(len).eq(0)
            | (events["time_precision"].eq("time") & events["window_hours"].le(0))
    )
    if invalid.any():
        bad_ids = events.loc[invalid, "event_id"].astype(str).tolist()
        raise ValueError(f"Invalid event calendar rows: {bad_ids}")

    # window_hours is a half-window only for timed events. For day-level
    # events it is retained for provenance but ignored by the validator.

    return events


def build_doge_price_daily2(
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


def build_doge_price_daily(
        start_date="2021-05-01",
        end_date="2021-05-23",
        output_file="doge_price_daily.csv"
):
    print(
        f"[INFO] Downloading DOGEUSDT daily klines "
        f"{start_date} -> {end_date} from Binance..."
    )

    start_ms = int(
        pd.Timestamp(start_date, tz="UTC").timestamp() * 1000
    )
    end_ms = int(
        pd.Timestamp(end_date, tz="UTC").timestamp() * 1000
    )

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "DOGEUSDT",
        "interval": "1d",
        "startTime": start_ms,
        "endTime": end_ms - 1,
        "limit": 1000,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        r = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        raise RuntimeError(
            f"Binance DOGEUSDT download failed: {e}"
        ) from e

    if not rows:
        raise RuntimeError(
            "Binance returned no DOGEUSDT data for the requested period."
        )

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(rows, columns=cols)

    df["day"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
        .dt.tz_convert(None)
        .dt.date
    )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["trades"] = pd.to_numeric(
        df["trades"], errors="coerce"
    ).astype("Int64")

    out = df[
        [
            "day",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
        ]
    ].copy()

    out["source"] = "Binance DOGEUSDT"

    start_day = pd.Timestamp(start_date).date()
    end_day = pd.Timestamp(end_date).date()
    out = out[
        (out["day"] >= start_day) &
        (out["day"] < end_day)
        ].sort_values("day").reset_index(drop=True)

    expected_days = pd.date_range(
        start=start_date,
        end=pd.Timestamp(end_date) - pd.Timedelta(days=1),
        freq="D",
    )
    found_days = pd.to_datetime(out["day"])
    missing_days = expected_days.difference(found_days)

    if len(missing_days) > 0:
        print(
            "[WARN] Missing DOGE price days:",
            ", ".join(
                d.strftime("%Y-%m-%d")
                for d in missing_days
            )
        )

    if out.empty:
        raise RuntimeError(
            "No DOGEUSDT daily data remained after date filtering."
        )

    out.to_csv(output_file, index=False)

    print(f"[OK] Saved: {output_file}")
    print(
        f"[OK] Price range: "
        f"{out['day'].min()} -> {out['day'].max()} "
        f"rows={len(out)}"
    )
    print(out.head())

    return out


def taxonomy_market_lag_correlation(
        workdir=WORKDIR,
        price_file=None,
        output_file=None,
):
    from scipy.stats import spearmanr

    if price_file is None:
        price_file = os.path.join(workdir, "doge_price_daily.csv")

    if output_file is None:
        output_file = os.path.join(
            workdir,
            "taxonomy_market_lag_correlation.csv"
        )

    scored_files = sorted([
        f for f in os.listdir(workdir)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    ])

    if not scored_files:
        raise RuntimeError(
            f"No scored_chunk_*.parquet files found under: {workdir}"
        )

    daily_parts = []

    wanted_taxonomies = ["T1", "T2", "T3", "T4", "T5", "T6"]

    for fn in scored_files:
        path = os.path.join(workdir, fn)

        df = pd.read_parquet(
            path,
            columns=[
                "date",
                "graph_id",
                "final_anomaly",
                "taxonomy_id",
                "taxonomy_components",
                "T1_match",
                "T2_match",
                "T3_match",
                "T4_match",
            ],
            engine="fastparquet",
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df["day"] = df["date"].dt.floor("D")

        total = (
            df.groupby("day")
            .size()
            .rename("total_tx")
            .reset_index()
        )

        anom = df[df["final_anomaly"] == 1].copy()

        anomaly_count = (
            anom.groupby("day")
            .size()
            .rename("anomaly_count")
            .reset_index()
        )

        if len(anom) > 0:
            tax_counts = (
                anom.groupby(["day", "taxonomy_id"])
                .size()
                .unstack(fill_value=0)
            )
        else:
            tax_counts = pd.DataFrame(index=total["day"])

        for t in wanted_taxonomies:
            if t not in tax_counts.columns:
                tax_counts[t] = 0

        tax_counts = (
            tax_counts[wanted_taxonomies]
            .reset_index()
        )

        component_counts = pd.DataFrame({"day": total["day"]})
        for t in ("T1", "T2", "T3", "T4"):
            flag_col = f"{t}_match"
            if flag_col in anom.columns:
                flag = anom[flag_col].fillna(False).astype(bool)
            else:
                flag = anom["taxonomy_components"].fillna("").map(
                    lambda value, target=t: target in {
                        x.strip() for x in str(value).split(",") if x.strip()
                    }
                )
            count = (
                anom.loc[flag]
                .groupby("day")
                .size()
                .rename(f"{t}_component_count")
                .reset_index()
            )
            component_counts = component_counts.merge(count, on="day", how="left")

        part = (
            total
            .merge(anomaly_count, on="day", how="left")
            .merge(tax_counts, on="day", how="left")
            .merge(component_counts, on="day", how="left")
        )

        component_cols = [f"{t}_component_count" for t in ("T1", "T2", "T3", "T4")]
        fill_cols = ["anomaly_count", *wanted_taxonomies, *component_cols]
        part[fill_cols] = part[fill_cols].fillna(0)

        daily_parts.append(part)

    daily = pd.concat(daily_parts, ignore_index=True)

    # Chunks can contain different portions of the same day,
    # so aggregate once more globally.
    agg_spec = {
        "total_tx": "sum",
        "anomaly_count": "sum",
    }
    for t in wanted_taxonomies:
        agg_spec[t] = "sum"
    for t in ("T1", "T2", "T3", "T4"):
        agg_spec[f"{t}_component_count"] = "sum"

    daily = (
        daily.groupby("day", as_index=False)
        .agg(agg_spec)
        .sort_values("day")
        .reset_index(drop=True)
    )

    # Rates relative to all transactions, not only anomalies.
    daily["overall_anomaly_rate"] = (
            daily["anomaly_count"] / daily["total_tx"]
    )

    for t in wanted_taxonomies:
        daily[f"final_{t}_rate"] = (
                daily[t] / daily["total_tx"]
        )

    for t in ("T1", "T2", "T3", "T4"):
        daily[f"{t}_component_rate"] = (
                daily[f"{t}_component_count"] / daily["total_tx"]
        )
    daily["T5_overlap_rate"] = daily["T5"] / daily["total_tx"]
    daily["T6_uncharacterized_rate"] = daily["T6"] / daily["total_tx"]

    if not os.path.exists(price_file):
        raise FileNotFoundError(
            f"Price file not found: {price_file}"
        )

    price = pd.read_csv(price_file)

    if "day" not in price.columns:
        raise RuntimeError(
            "doge_price_daily.csv must contain a 'day' column."
        )

    if "close" not in price.columns:
        raise RuntimeError(
            "doge_price_daily.csv must contain a 'close' column."
        )

    price["day"] = pd.to_datetime(
        price["day"],
        errors="coerce"
    ).dt.floor("D")

    price["close"] = pd.to_numeric(
        price["close"],
        errors="coerce"
    )

    price = (
        price.dropna(subset=["day", "close"])
        .sort_values("day")
        .drop_duplicates("day")
        .reset_index(drop=True)
    )

    price["return"] = price["close"].pct_change()
    price["abs_return"] = price["return"].abs()

    merged = (
        daily.merge(
            price[["day", "close", "return", "abs_return"]],
            on="day",
            how="inner",
        )
        .sort_values("day")
        .reset_index(drop=True)
    )

    merged_path = os.path.join(
        workdir,
        "taxonomy_market_daily_merged.csv"
    )
    merged.to_csv(merged_path, index=False)

    print(
        f"[OK] Daily blockchain/market table saved: {merged_path}"
    )

    def safe_spearman(x, y):
        pair = pd.DataFrame({
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }).dropna()

        if len(pair) < 3:
            return np.nan, np.nan, len(pair)

        if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
            return np.nan, np.nan, len(pair)

        rho, pvalue = spearmanr(
            pair["x"],
            pair["y"]
        )

        return float(rho), float(pvalue), len(pair)

    signals = [
        "overall_anomaly_rate",
        "T1_component_rate",
        "T2_component_rate",
        "T3_component_rate",
        "T4_component_rate",
        "T5_overlap_rate",
        "T6_uncharacterized_rate",
    ]

    results = []

    for signal in signals:
        rho, pvalue, n = safe_spearman(
            merged[signal],
            merged["abs_return"]
        )

        results.append({
            "signal": signal,
            "alignment": "same_day",
            "lag_days": 0,
            "spearman_rho": rho,
            "p_value": pvalue,
            "n_days": n,
            "x_definition": f"{signal}_t",
            "y_definition": "|return_t|",
        })

        rho, pvalue, n = safe_spearman(
            merged[signal],
            merged["abs_return"].shift(-1)
        )

        results.append({
            "signal": signal,
            "alignment": "anomaly_to_price_plus1d",
            "lag_days": 1,
            "spearman_rho": rho,
            "p_value": pvalue,
            "n_days": n,
            "x_definition": f"{signal}_t",
            "y_definition": "|return_(t+1)|",
        })

        rho, pvalue, n = safe_spearman(
            merged["abs_return"],
            merged[signal].shift(-1)
        )

        results.append({
            "signal": signal,
            "alignment": "price_to_anomaly_plus1d",
            "lag_days": 1,
            "spearman_rho": rho,
            "p_value": pvalue,
            "n_days": n,
            "x_definition": "|return_t|",
            "y_definition": f"{signal}_(t+1)",
        })

    result_df = pd.DataFrame(results)

    # Control the false-discovery rate across all taxonomy/alignment tests.
    p = pd.to_numeric(result_df["p_value"], errors="coerce")
    valid = p.notna()
    q = pd.Series(np.nan, index=result_df.index, dtype=float)
    if valid.any():
        ordered = p.loc[valid].sort_values()
        m = len(ordered)
        adjusted = ordered.to_numpy() * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        q.loc[ordered.index] = np.clip(adjusted, 0.0, 1.0)
    result_df["fdr_bh_q_value"] = q
    result_df["significant_fdr_q_lt_0_05"] = result_df["fdr_bh_q_value"] < 0.05

    def strength_label(r):
        if pd.isna(r):
            return "NA"
        a = abs(r)
        if a < 0.10:
            return "negligible"
        if a < 0.30:
            return "weak"
        if a < 0.50:
            return "moderate"
        return "strong"

    result_df["strength"] = (
        result_df["spearman_rho"]
        .apply(strength_label)
    )

    result_df["direction"] = np.where(
        result_df["spearman_rho"] > 0,
        "positive",
        np.where(
            result_df["spearman_rho"] < 0,
            "negative",
            "zero"
        )
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n[OK] Taxonomy-market lag correlations saved: "
        f"{output_file}\n"
    )

    display_cols = [
        "signal",
        "alignment",
        "spearman_rho",
        "p_value",
        "fdr_bh_q_value",
        "n_days",
        "strength",
        "direction",
    ]

    print(
        result_df[display_cols]
        .sort_values(
            ["alignment", "spearman_rho"],
            ascending=[True, False]
        )
        .to_string(index=False)
    )

    return result_df, merged


def _audit_direct_taxonomy(label_filter=None):
    """Audit saved direct-profile labels without changing them."""
    scored_files = sorted(
        f for f in os.listdir(WORKDIR)
        if f.startswith("scored_chunk_") and f.endswith(".parquet")
    )
    if not scored_files:
        raise RuntimeError(f"No scored_chunk_*.parquet files in {WORKDIR}")

    detail_parts = []
    combination_counts = Counter()
    candidate_counts = Counter()

    columns = [
        "graph_id", "date", "final_anomaly", "taxonomy_id",
        "taxonomy_components", "taxonomy_n_components",
        *[f"{tid}_match" for tid in ("T1", "T2", "T3", "T4")],
    ]

    for fn in scored_files:
        df = pd.read_parquet(
            os.path.join(WORKDIR, fn), columns=columns, engine="fastparquet"
        )
        df = df.loc[pd.to_numeric(df["final_anomaly"]).eq(1)].copy()
        if label_filter is not None:
            df = df.loc[df["taxonomy_id"].isin(label_filter)]
        if df.empty:
            continue

        flags = df[[f"{tid}_match" for tid in ("T1", "T2", "T3", "T4")]].astype(bool)
        n_components = flags.sum(axis=1)
        expected = pd.Series("T6", index=df.index, dtype="object")
        expected.loc[n_components >= 2] = "T5"
        for tid in ("T1", "T2", "T3", "T4"):
            expected.loc[(n_components == 1) & flags[f"{tid}_match"]] = tid

        components = flags.apply(
            lambda row: ",".join(
                tid for tid in ("T1", "T2", "T3", "T4")
                if bool(row[f"{tid}_match"])
            ),
            axis=1,
        )
        part = df[["graph_id", "date", "taxonomy_id"]].copy()
        for tid in ("T1", "T2", "T3", "T4"):
            part[f"{tid}_match"] = flags[f"{tid}_match"].astype("int8")
        part["candidate_taxonomies"] = components
        part["n_candidate_taxonomies"] = n_components.astype("int8")
        part["expected_direct_label"] = expected
        part["label_matches_rule"] = part["taxonomy_id"].eq(expected)
        detail_parts.append(part)

        for taxonomy_id, component_set in zip(df["taxonomy_id"], components):
            combination_counts[(str(taxonomy_id), component_set or "none")] += 1
            for tid in filter(None, component_set.split(",")):
                candidate_counts[(str(taxonomy_id), tid)] += 1

    if not detail_parts:
        raise RuntimeError("No eligible anomalous rows were found for taxonomy audit")

    detail = pd.concat(detail_parts, ignore_index=True)
    if label_filter == {"T5"}:
        prefix = "t5_semantic_overlap"
    else:
        prefix = "taxonomy_semantic_overlap"
    detail_path = os.path.join(WORKDIR, f"{prefix}_detail.csv")
    combinations_path = os.path.join(WORKDIR, f"{prefix}_combinations.csv")
    candidates_path = os.path.join(WORKDIR, f"{prefix}_candidates.csv")
    detail.to_csv(detail_path, index=False)
    pd.DataFrame([
        {
            "rule_version": PROFILE_RULE_VERSION,
            "taxonomy_id": taxonomy_id,
            "candidate_taxonomies": component_set,
            "count": count,
        }
        for (taxonomy_id, component_set), count in sorted(combination_counts.items())
    ]).to_csv(combinations_path, index=False)
    pd.DataFrame([
        {
            "rule_version": PROFILE_RULE_VERSION,
            "taxonomy_id": taxonomy_id,
            "candidate_taxonomy": tid,
            "count": count,
        }
        for (taxonomy_id, tid), count in sorted(candidate_counts.items())
    ]).to_csv(candidates_path, index=False)

    mismatches = int((~detail["label_matches_rule"]).sum())
    if mismatches:
        raise RuntimeError(f"Direct taxonomy audit failed: {mismatches} label mismatches")
    print(f"[OK] Direct taxonomy audit: {len(detail)} rows; rule mismatches=0")
    return detail


def audit_t5_semantic_overlap():
    """Audit component combinations of saved T5 composite profiles."""
    return _audit_direct_taxonomy(label_filter={"T5"})


def audit_all_semantic_overlap():
    """Audit all saved T1--T6 labels against their component flags."""
    return _audit_direct_taxonomy()


def main():
    print("\n[INFO] Starting leakage-free IF -> direct taxonomy -> event validation pipeline...\n")

    # Rebuild centralities and features after graph changes:
    rebuild_features_with_train_centrality()

    # Refit IF, calibrate its threshold, and assign profiles:
    train_if_and_profile_taxonomy()

    # Optional reporting/validation stages:
    print_final_taxonomy_summary()
    print_daily_taxonomy_distribution()
    event_comparison_validation()
    audit_all_semantic_overlap()
    build_doge_price_daily()  # run once when the daily price file is absent
    taxonomy_market_lag_correlation()

    print("\n[OK] Leakage-free IF + direct T1--T6 taxonomy + event validation completed.\n")


if __name__ == "__main__":
    main()