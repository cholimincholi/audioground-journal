#!/usr/bin/env python
"""Stage 3: clip pools.

* background pool: AudioCaps-train clips whose AudioSet weak labels are ALL stationary-ambience classes
  (wind / rain / water / natural outdoor / noise). These clips are removed from the foreground pool so a
  background bed never re-appears as a query target. (No Clotho, no walking-tour audio.)
* strong events: for AudioCaps clips that appear in AudioSet-Strong (train/eval TSVs), keep the events and
  mark which strong labels match the caption (SBERT cosine between caption and label display name).

Outputs <out>/pools.parquet (row-aligned with emb.npy) and <out>/strong_events.parquet.
"""
import argparse, json, os, re
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

AMBIENCE = {"Wind", "Rustling leaves", "Rain", "Raindrop", "Rain on surface", "Stream", "Waves, surf", "Ocean",
            "Environmental noise", "White noise", "Pink noise", "Wind noise (microphone)", "Water", "Waterfall",
            "Outside, rural or natural", "Hum", "Mains hum", "Air conditioning", "Mechanical fan", "Noise",
            "Static", "Silence"}
# caption words that disqualify a clip as background (weak labels are incomplete)
BAD_WORDS = (r"\b(speak\w*|speech|talk\w*|man|men|woman|women|person|people|child\w*|kid\w*|bab(y|ies)|"
             r"voice\w*|laugh\w*|music\w*|sing\w*|song|engine\w*|car|cars|vehicle\w*|truck\w*|motor\w*|"
             r"boat\w*|plane|aircraft|train|dog\w*|bark\w*|frog\w*|bird\w*|duck\w*|quack\w*|siren\w*|horn\w*|"
             r"bell\w*|gun\w*|shot\w*|explo\w*|thunder\w*|clap\w*|beep\w*|whistl\w*|knock\w*|bang\w*|"
             r"insect\w*|guitar\w*|vocal\w*|human\w*|click\w*|camera\w*|object\w*|bee|bees|buzz\w*|cricket\w*|animal\w*|footstep\w*|walk\w*|splash\w*)\b")
LABEL_MATCH_TAU = 0.30   # caption <-> strong-label-name cosine to count as "caption-relevant"


def read_weak(d):
    rows = []
    for f in ["balanced_train_segments.csv", "unbalanced_train_segments.csv", "eval_segments.csv"]:
        for line in open(os.path.join(d, f)):
            if line.startswith("#"):
                continue
            p = line.strip().split(", ", 3)
            rows.append((p[0], int(float(p[1])), p[3].strip('"')))
    return pd.DataFrame(rows, columns=["youtube_id", "weak_start", "weak_mids"]).drop_duplicates("youtube_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    R = a.root
    clips = pd.read_parquet(f"{R}/clips/clips.parquet")
    idx = pd.read_parquet(f"{R}/emb/emb_index.parquet")
    df = idx.merge(clips, on="audiocap_id", how="left")
    assert len(df) == len(idx) and df.keep.all()

    weak = read_weak(f"{R}/raw/audioset_weak")
    cl = pd.read_csv(f"{R}/raw/audioset_weak/class_labels_indices.csv")
    m2n = dict(zip(cl.mid, cl.display_name))
    df = df.merge(weak, on="youtube_id", how="left")
    df["weak_labels"] = df.weak_mids.fillna("").apply(lambda s: [m2n.get(m, m) for m in s.split(",") if m])
    bad = re.compile(BAD_WORDS, re.I)
    df["is_background"] = df.weak_labels.apply(lambda l: len(l) > 0 and set(l) <= AMBIENCE) & \
        ~df.caption.str.contains(bad)

    # ---------------- strong labels
    s2n = pd.read_csv(f"{R}/raw/audioset_strong/mid_to_display_name.tsv", sep="\t", header=None,
                      names=["mid", "name"])
    s2n = dict(zip(s2n.mid, s2n.name))
    st = pd.concat([pd.read_csv(f"{R}/raw/audioset_strong/audioset_train_strong.tsv", sep="\t").assign(src="train"),
                    pd.read_csv(f"{R}/raw/audioset_strong/audioset_eval_strong.tsv", sep="\t").assign(src="eval")])
    st["name"] = st.label.map(s2n).fillna(st.label)
    df["segment_id"] = df.youtube_id + "_" + (df.start_time * 1000).astype(int).astype(str)
    st = st[st.segment_id.isin(set(df.segment_id))].copy()
    m = SentenceTransformer(a.model, device="cpu")
    names = sorted(st.name.unique())
    En = m.encode(names, normalize_embeddings=True, convert_to_numpy=True)
    E = np.load(f"{R}/emb/emb.npy")
    row_of = dict(zip(df.segment_id, range(len(df))))
    nidx = {n: i for i, n in enumerate(names)}
    st["row"] = st.segment_id.map(row_of)
    st["cap_sim"] = np.einsum("ij,ij->i", E[st.row.values], En[st.name.map(nidx).values])
    st["matched"] = st.cap_sim >= LABEL_MATCH_TAU
    # if no label passes the threshold for a clip, fall back to its single best-matching label (if >=0.2)
    best = st.groupby("row").cap_sim.transform("max")
    nomatch = ~st.groupby("row").matched.transform("any")
    st.loc[nomatch & (st.cap_sim == best) & (st.cap_sim >= 0.20), "matched"] = True
    st = st.rename(columns={"start_time_seconds": "on", "end_time_seconds": "off"})
    st[["row", "segment_id", "on", "off", "label", "name", "cap_sim", "matched", "src"]].reset_index(drop=True) \
        .to_parquet(f"{R}/pools/strong_events.parquet")
    df["has_strong"] = df.index.isin(set(st.row))
    df["has_strong_matched"] = df.index.isin(set(st[st.matched].row))
    os.makedirs(f"{R}/pools", exist_ok=True)
    df[["audiocap_id", "youtube_id", "start_time", "caption", "offset", "n_samples", "dur", "is_background",
        "has_strong", "has_strong_matched", "weak_labels"]].to_parquet(f"{R}/pools/pools.parquet")
    allc = clips
    rep = dict(
        audiocaps_train_official=int(len(allc)), with_audio=int(allc.has_audio.sum()), kept=int(len(df)),
        background_pool=int(df.is_background.sum()),
        strong_cov_official=int(allc.assign(seg=allc.youtube_id + "_" + (allc.start_time * 1000).astype(int).astype(str))
                                .seg.isin(set(pd.read_csv(f"{R}/raw/audioset_strong/audioset_train_strong.tsv", sep="\t").segment_id)
                                          | set(pd.read_csv(f"{R}/raw/audioset_strong/audioset_eval_strong.tsv", sep="\t").segment_id)).sum()),
        strong_cov_kept=int(df.has_strong.sum()), strong_cov_kept_matched=int(df.has_strong_matched.sum()),
        strong_events_on_kept=int(len(st)),
    )
    json.dump(rep, open(f"{R}/pools/pools_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
