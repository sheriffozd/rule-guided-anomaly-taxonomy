#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import csv, pandas as pd

WORKDIR=Path("./work_tx_address_gds13_train_centrality")
LABELS=("T1","T2","T3","T4","T5","T6")
COMPONENTS=("T1","T2","T3","T4")

def main():
    files=sorted(WORKDIR.glob("scored_chunk_*.parquet"))
    if not files: raise RuntimeError("No scored chunks; run file 1 first")
    stats=Counter()
    for path in files:
        cols=["final_anomaly","taxonomy_id","taxonomy_components","taxonomy_n_components",*[f"{t}_match" for t in COMPONENTS]]
        df=pd.read_parquet(path,columns=cols); df=df[pd.to_numeric(df.final_anomaly).eq(1)]
        bad=set(df.taxonomy_id)-set(LABELS)
        if bad: raise ValueError(f"Unexpected labels: {sorted(bad)}")
        n=df[[f"{t}_match" for t in COMPONENTS]].astype(bool).sum(axis=1)
        expected=pd.Series("T6",index=df.index); expected[n>=2]="T5"
        for t in COMPONENTS: expected[(n==1)&df[f"{t}_match"].astype(bool)]=t
        stats["anomalies"]+=len(df); stats["label_mismatch"]+=int((expected!=df.taxonomy_id).sum())
        stats["component_count_mismatch"]+=int((n!=pd.to_numeric(df.taxonomy_n_components)).sum())
        for t in LABELS: stats[f"label:{t}"]+=int((df.taxonomy_id==t).sum())
        for (label,components),count in df.groupby(["taxonomy_id","taxonomy_components"],dropna=False).size().items():
            stats[f"combination:{label}:{components}"]+=int(count)
    output=WORKDIR/"taxonomy_overlap_audit.csv"
    with output.open("w",newline="",encoding="utf-8") as h:
        w=csv.writer(h); w.writerow(["metric","label","components","count","denominator","rate_pct"])
        total=stats["anomalies"]
        w.writerow(["anomalies","","",total,total,100 if total else ""])
        for key in ("label_mismatch","component_count_mismatch"):
            n=stats[key]; w.writerow([key,"","",n,total,100*n/total if total else ""])
        for t in LABELS:
            n=stats[f"label:{t}"]; w.writerow(["final_label",t,"",n,total,100*n/total if total else ""])
        for key,n in sorted(stats.items()):
            if key.startswith("combination:"):
                _,label,components=key.split(":",2); w.writerow(["component_combination",label,components,n,total,100*n/total if total else ""])
    if stats["label_mismatch"] or stats["component_count_mismatch"]: raise RuntimeError(f"Taxonomy audit failed: {dict(stats)}")
    print(f"Saved {output}; anomalies={total}; rule mismatches=0")

if __name__=="__main__": main()
