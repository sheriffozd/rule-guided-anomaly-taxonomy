#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd

WORKDIR=Path("./resources/work_tx_address_gds13_train_centrality")
OUTDIR=WORKDIR/"detector_taxonomy_robustness"
PROFILE_IDS=("T1","T2","T3","T4","T5","T6")
COMPONENTS=("T1","T2","T3","T4")
FEATURES=["value","tx_count_15min_z","active_sender_15m_log",
          *[f"{side}_{metric}_log" for side in ("sender","receiver")
            for metric in ("degree","pageRank","eigenvector")]]
LABEL_CANDIDATES={"IF":["benchmark_if_label","if_label","anomaly_label_gds"],
                  "LOF":["lof_label"],"OCSVM":["ocsvm_label"],
                  "COPOD":["copod_label"],"ECOD":["ecod_label"]}

def flags(frame,t):
    x=frame[FEATURES].apply(pd.to_numeric,errors="coerce")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite taxonomy features")
    f=pd.DataFrame(index=frame.index)
    f["T1"]=(x.tx_count_15min_z>=t["density"])&(x.tx_count_15min_z>0)
    f["T2"]=(x.value>=t["value"])&(x.value>0)
    f["T3"]=(x.active_sender_15m_log>=t["senders"])&(x.active_sender_15m_log>0)
    endpoint=[]
    for side in ("sender","receiver"):
        votes=sum(((x[f"{side}_{m}_log"]>=t[m])&(x[f"{side}_{m}_log"]>0)).astype("int8")
                  for m in ("degree","pageRank","eigenvector"))
        endpoint.append(votes>=2)
    f["T4"]=endpoint[0]|endpoint[1]
    return f

def classify(frame,label,t):
    anomaly=pd.to_numeric(frame[label],errors="raise").eq(1)
    result=pd.Series("NORMAL",index=frame.index,dtype=object)
    components=pd.Series("",index=frame.index,dtype=object)
    if anomaly.any():
        f=flags(frame.loc[anomaly],t); n=f.sum(axis=1)
        ids=pd.Series("T6",index=f.index,dtype=object); ids.loc[n>=2]="T5"
        for tid in COMPONENTS: ids.loc[(n==1)&f[tid]]=tid
        bits=sum(f[tid].astype("int8")*(1<<i) for i,tid in enumerate(COMPONENTS))
        lookup={b:",".join(tid for i,tid in enumerate(COMPONENTS) if b&(1<<i)) for b in range(16)}
        result.loc[anomaly]=ids; components.loc[anomaly]=bits.map(lookup)
    return result,components

def main():
    files=sorted(WORKDIR.glob("baseline_scored_chunk_*.parquet"))
    if not files: raise RuntimeError("Run file 2 first; baseline chunks are missing")
    with (WORKDIR/"profile_thresholds.json").open() as h: meta=json.load(h)
    if meta.get("rule_version")!="behavior_profiles_v1": raise ValueError("Rule-version mismatch")
    sample_cols=set(pd.read_parquet(files[0]).columns)
    labels={d:next((c for c in cs if c in sample_cols),None) for d,cs in LABEL_CANDIDATES.items()}
    labels={d:c for d,c in labels.items() if c}
    overall=[]; daily=[]; split=[]; combos=[]; sensitivity=[]
    scenarios=meta["scenarios"]
    for path in files:
        cols=["date","dataset_split",*FEATURES,*labels.values()]
        df=pd.read_parquet(path,columns=list(dict.fromkeys(cols)))
        day=pd.to_datetime(df.date,errors="coerce",utc=True).dt.strftime("%Y-%m-%d")
        for detector,label in labels.items():
            base,comp=classify(df,label,scenarios["baseline"])
            anomaly=base.ne("NORMAL")
            for tid in PROFILE_IDS:
                n=int((base==tid).sum()); overall.append((detector,tid,n,int(anomaly.sum())))
            temp=pd.DataFrame({"day":day,"taxonomy_id":base})[anomaly]
            for (d,t),n in temp.groupby(["day","taxonomy_id"]).size().items(): daily.append((detector,d,t,int(n)))
            temp=pd.DataFrame({"split":df.dataset_split,"taxonomy_id":base})[anomaly]
            for (sp,t),n in temp.groupby(["split","taxonomy_id"]).size().items(): split.append((detector,sp,t,int(n)))
            for (tid,c),n in pd.DataFrame({"taxonomy_id":base,"components":comp})[anomaly].groupby(["taxonomy_id","components"]).size().items():
                combos.append((detector,tid,c,int(n)))
            for scenario,t in scenarios.items():
                ids,_=classify(df,label,t)
                for tid in PROFILE_IDS: sensitivity.append((detector,scenario,tid,int((ids==tid).sum())))
    OUTDIR.mkdir(parents=True,exist_ok=True)
    o=pd.DataFrame(overall,columns=["detector","taxonomy_id","count","anomalies"]).groupby(["detector","taxonomy_id"],as_index=False).agg({"count":"sum","anomalies":"sum"})
    o["percentage"]=100*o["count"]/o["anomalies"].replace(0,np.nan); o.to_csv(OUTDIR/"detector_taxonomy_summary.csv",index=False)
    pd.DataFrame(daily,columns=["detector","day","taxonomy_id","count"]).groupby(["detector","day","taxonomy_id"],as_index=False)["count"].sum().to_csv(OUTDIR/"detector_daily_taxonomy_counts.csv",index=False)
    pd.DataFrame(split,columns=["detector","dataset_split","taxonomy_id","count"]).groupby(["detector","dataset_split","taxonomy_id"],as_index=False)["count"].sum().to_csv(OUTDIR/"detector_split_taxonomy_summary.csv",index=False)
    pd.DataFrame(combos,columns=["detector","taxonomy_id","taxonomy_components","count"]).groupby(["detector","taxonomy_id","taxonomy_components"],as_index=False)["count"].sum().to_csv(OUTDIR/"detector_taxonomy_components.csv",index=False)
    pd.DataFrame(sensitivity,columns=["detector","scenario","taxonomy_id","count"]).groupby(["detector","scenario","taxonomy_id"],as_index=False)["count"].sum().to_csv(OUTDIR/"threshold_sensitivity_taxonomy_summary.csv",index=False)
    print(o.to_string(index=False))

if __name__=="__main__": main()
