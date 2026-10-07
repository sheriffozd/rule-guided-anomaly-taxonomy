#!/usr/bin/env python3
import getpass
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

WORKDIR = Path(
    '/home/serife/PycharmProjects/SerifeTez/tez/v3_new_taxonomy_2/'
    'work_tx_address_gds13_train_centrality'
)
OUTDIR = WORKDIR / 't4_address_concentration'
TRAIN_END = '2021-05-17'
TEST_START = '2021-05-18'
METRICS = ('degree', 'pageRank', 'eigenvector')
FEATURES = [f'{side}_{metric}_log' for side in ('sender', 'receiver') for metric in METRICS]
BATCH_SIZE = 1000
TOP_K = (1, 10, 100)


QUERY = '''
UNWIND $graph_ids AS gid
MATCH (t:RawTransaction {graph_id: gid})
CALL {
  WITH t
  OPTIONAL MATCH (tx:RawTransaction {id: t.id})
  OPTIONAL MATCH (a:Address)-[:SENT]->(tx)
  RETURN collect(DISTINCT CASE WHEN a IS NULL THEN null ELSE {
    address: a.address, degree: coalesce(a.degree_train, 0.0),
    pageRank: coalesce(a.pageRank_train, 0.0),
    eigenvector: coalesce(a.eigenvector_train, 0.0)
  } END) AS sender_candidates
}
CALL {
  WITH t
  OPTIONAL MATCH (tx:RawTransaction {id: t.id})
  OPTIONAL MATCH (tx)-[:RECEIVED]->(a:Address)
  RETURN collect(DISTINCT CASE WHEN a IS NULL THEN null ELSE {
    address: a.address, degree: coalesce(a.degree_train, 0.0),
    pageRank: coalesce(a.pageRank_train, 0.0),
    eigenvector: coalesce(a.eigenvector_train, 0.0)
  } END) AS receiver_candidates
}
RETURN t.graph_id AS graph_id, sender_candidates, receiver_candidates
'''


def load_thresholds(path):
    raw = path.read_bytes()
    metadata = json.loads(raw)
    if metadata.get('rule_version') != 'behavior_profiles_v1':
        raise ValueError('Expected behavior_profiles_v1 thresholds')
    if metadata.get('train_end_exclusive') != TRAIN_END or metadata.get('baseline_quantile') != .95:
        raise ValueError('Expected the current TRAIN-only baseline P95 thresholds')
    t = {metric: float(metadata['thresholds'][metric]) for metric in METRICS}
    if not all(np.isfinite(v) and v >= 0 for v in t.values()):
        raise ValueError('Invalid centrality thresholds')
    return t, hashlib.sha256(raw).hexdigest()


def select_t4(df, thresholds):
    labels = pd.to_numeric(df['final_anomaly'], errors='raise')
    if not labels.isin([0, 1]).all():
        raise ValueError('final_anomaly must be 0/1')
    x = df.loc[labels.eq(1)].copy()
    numeric = x[FEATURES].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('Non-finite endpoint features')
    for side in ('sender', 'receiver'):
        votes = sum(((numeric[f'{side}_{metric}_log'] >= thresholds[metric]) &
                     (numeric[f'{side}_{metric}_log'] > 0)).astype('int8') for metric in METRICS)
        x[f'{side}_qualifies'] = votes >= 2
    x = x.loc[x.sender_qualifies | x.receiver_qualifies].copy()
    x[FEATURES] = numeric.loc[x.index]
    date = pd.to_datetime(x['date'], utc=True, errors='coerce')
    if date.isna().any():
        raise ValueError('Invalid date in a T4 anomaly record')
    x['day'] = date.dt.strftime('%Y-%m-%d')
    x['dataset_split'] = np.where(x.day < TRAIN_END, 'TRAIN',
                                  np.where(x.day < TEST_START, 'VALID', 'TEST'))
    ids = pd.to_numeric(x['graph_id'], errors='raise')
    if ids.isna().any() or not np.isfinite(ids.to_numpy(dtype=float)).all() or (ids % 1 != 0).any():
        raise ValueError('Invalid graph_id')
    x['graph_id'] = ids.astype('int64')
    if x.graph_id.duplicated().any():
        raise ValueError('Duplicate T4 graph_id within input chunk')
    return x


def resolve_address(candidates, row, side):
    expected = np.array([row[f'{side}_{metric}_log'] for metric in METRICS], dtype=np.float32)
    matches = set()
    for candidate in candidates:

        raw = np.array([candidate[metric] for metric in METRICS], dtype=np.float32)
        actual = np.log1p(np.maximum(raw, np.float32(0))).astype(np.float32)


        if np.array_equal(expected, actual):
            address = candidate.get('address')
            if isinstance(address, str) and address.strip():
                matches.add(address)
    if len(matches) != 1:
        raise ValueError(f"graph_id={row['graph_id']}, {side}: {len(matches)} matching address "
                         'identities. The DB snapshot may differ or the original head(collect(...)) '
                         'lookup may be ambiguous. Do not guess an address; export endpoint '
                         'identities with the original features before continuing.')
    return next(iter(matches))


def fetch_endpoints(session, batch):
    ids = [int(v) for v in batch.graph_id]
    found = {}
    for record in session.run(QUERY, graph_ids=ids):
        gid = int(record['graph_id'])
        if gid in found:
            raise ValueError(f'graph_id={gid} is not unique in Neo4j')
        found[gid] = dict(record)
    missing = set(ids) - set(found)
    if missing:
        raise ValueError(f'{len(missing)} T4 records missing from Neo4j; example={min(missing)}')
    resolved = []
    for _, row in batch.iterrows():
        gid = int(row.graph_id)
        node = found[gid]
        sender = resolve_address(node['sender_candidates'], row, 'sender') if row.sender_qualifies else None
        receiver = resolve_address(node['receiver_candidates'], row, 'receiver') if row.receiver_qualifies else None
        resolved.append((gid, row.dataset_split, sender, receiver))
    return resolved


class RecordStore:

    def __init__(self):
        self.records = {}

    def add(self, rows):
        for gid, split, sender, receiver in rows:
            if gid in self.records:
                raise ValueError('Duplicate T4 graph_id across chunks; input files must belong to one run')
            self.records[gid] = (split, sender, receiver)


def summarize_scope(store, scope):
    records = [r for r in store.records.values() if scope == 'ALL' or r[0] == scope]
    total = len(records)
    counts = Counter()
    for _, sender, receiver in records:
        counts.update({address for address in (sender, receiver) if address is not None})
    rankings = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    summary = dict(period=scope, t4_anomaly_records=total, qualified_unique_addresses=len(rankings))
    for k in TOP_K:
        selected = {address for address, _ in rankings[:k]}
        covered = sum(sender in selected or receiver in selected for _, sender, receiver in records)
        summary[f'top_{k}_actual_addresses'] = min(k, len(rankings))
        summary[f'top_{k}_covered_records'] = covered
        summary[f'top_{k}_coverage_percent'] = 100. * covered / total if total else np.nan
    ranked_rows = [dict(period=scope, rank=i, address=address, t4_record_count=n,
                        t4_record_coverage_percent=100.*n/total) for i, (address,n) in enumerate(rankings,1)]
    return summary, ranked_rows


def latex_table(summary):
    lines = [r'\begin{table}[!t]', r'\caption{Address Concentration Among Anomalies Satisfying T4}',
             r'\label{tab:t4_address_concentration}', r'\centering', r'\scriptsize',
             r'\begin{tabular}{lrrrr}', r'\hline',
             r'Period & Addresses & Top 1 (\%) & Top 10 (\%) & Top 100 (\%) \\', r'\hline']
    for scope, label in [('ALL', 'Overall'), ('TEST', 'Test')]:
        row = summary.loc[summary.period.eq(scope)].iloc[0]
        numbers = ['--' if pd.isna(row[f'top_{k}_coverage_percent']) else
                   f"{row[f'top_{k}_coverage_percent']:.3f}" for k in TOP_K]
        lines.append(f"{label} & {int(row.qualified_unique_addresses):,} & " + ' & '.join(numbers) + r' \\')
    lines += [r'\hline', r'\end{tabular}', r'\par\vspace{2pt}', r'\begin{minipage}{\columnwidth}\scriptsize',
              'Only endpoints satisfying the T4 centrality rule are ranked. T4 components '
              'within T5 are included. Coverage is the percentage of T4 anomalous records '
              'containing at least one of the selected qualifying addresses; each record '
              'is counted once. Addresses are ranked separately within each period. '
              'If fewer than k addresses exist, Top k includes all available addresses. '
              'The unit is a derived input--output association record.',
              r'\end{minipage}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def main():
    from neo4j import GraphDatabase
    thresholds, digest = load_thresholds(WORKDIR / 'profile_thresholds.json')
    files = sorted(WORKDIR.glob('base_scored_chunk_*.parquet'))
    if not files:
        raise FileNotFoundError(f'No base_scored_chunk files in {WORKDIR.resolve()}')
    password = os.environ.get('NEO4J_PASSWORD') or getpass.getpass('Neo4j password: ')
    if not password:
        raise ValueError('Neo4j password is required')
    uri = os.environ.get('NEO4J_URI', 'bolt://127.0.0.1:7688')
    user = os.environ.get('NEO4J_USER', 'neo4j')
    driver = GraphDatabase.driver(uri, auth=(user, password))
    manifest, total_rows, anomaly_rows = [], 0, 0
    store = RecordStore()
    try:
        driver.verify_connectivity()
        with driver.session(default_access_mode='READ') as session:
            for i, path in enumerate(files, 1):
                df = pd.read_parquet(path, columns=['graph_id', 'date', 'final_anomaly', *FEATURES])
                total_rows += len(df)
                t4 = select_t4(df, thresholds)
                anomaly_rows += int(pd.to_numeric(df.final_anomaly).eq(1).sum())
                del df
                for start in range(0, len(t4), BATCH_SIZE):
                    batch = t4.iloc[start:start+BATCH_SIZE]
                    resolved = fetch_endpoints(session, batch)
                    store.add(resolved)
                manifest.append(dict(name=path.name, t4_records=len(t4), bytes=path.stat().st_size))
                print(f'[{i}/{len(files)}] {path.name}: T4 anomaly records={len(t4):,}', flush=True)
        summaries, rankings = [], []
        for scope in ('ALL', 'TRAIN', 'VALID', 'TEST'):
            row, ranks = summarize_scope(store, scope)
            summaries.append(row); rankings.extend(ranks)
        summary = pd.DataFrame(summaries)
        OUTDIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUTDIR / 't4_address_concentration_summary.csv', index=False,
                       float_format='%.8f', na_rep='NA')
        pd.DataFrame(rankings, columns=['period','rank','address','t4_record_count',
            't4_record_coverage_percent']).to_csv(OUTDIR / 't4_address_rankings.csv', index=False)
        (OUTDIR / 't4_address_concentration_table.tex').write_text(latex_table(summary), encoding='utf-8')
        notes = dict(unit='derived input-output association record', thresholds=thresholds,
                     threshold_file_sha256=digest, total_input_records=total_rows,
                     total_anomalies=anomaly_rows, input_manifest=manifest,
                     train_end_exclusive=TRAIN_END, test_start_inclusive=TEST_START,
                     address_source='Neo4j original id-based export lookup, checked against stored float32 log features',
                     tie_break='record count descending, address ascending',
                     limitations=['No txid-level deduplication or wallet/entity clustering.',
                                  'ALL and TEST ranks are independent; this is descriptive concentration.',
                                  'No claim about ownership, fraud, bots, or exchange identities.'])
        (OUTDIR / 't4_address_concentration_notes.json').write_text(
            json.dumps(notes, indent=2, allow_nan=False), encoding='utf-8')
        print('\n' + summary.to_string(index=False))
        print(f'\nSaved: {OUTDIR.resolve()}')
    finally:
        driver.close()


if __name__ == '__main__':
    main()
