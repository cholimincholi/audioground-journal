#!/usr/bin/env python
"""Stage 2: Sentence-BERT embeddings of AudioCaps train captions (CPU) + pairwise-similarity stats.

--compare  : embed with several SBERT models, print random-pair cosine stats for each (model selection).
default    : embed with --model, save emb.npy (L2-normalized, row-aligned with kept clips), stats json,
             and top-K neighbour table (for composing long target spans from similar clips).
"""
import argparse, json, os
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(int(os.environ.get("NTHREADS", "32")))


def pair_stats(E, n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(E), n); j = rng.integers(0, len(E), n)
    m = i != j
    s = np.einsum("ij,ij->i", E[i[m]], E[j[m]])
    q = np.percentile(s, [5, 25, 50, 75, 95])
    return dict(mean=float(s.mean()), std=float(s.std()), p5=q[0], p25=q[1], p50=q[2], p75=q[3], p95=q[4],
                frac_below={str(t): float((s < t).mean()) for t in (0.25, 0.33, 0.41)}), s


def embed(model_name, caps, bs=256):
    m = SentenceTransformer(model_name, device="cpu")
    return m.encode(caps, batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
                    convert_to_numpy=True).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--compare", nargs="*")
    ap.add_argument("--topk", type=int, default=200)
    a = ap.parse_args()
    df = pd.read_parquet(a.clips)
    df = df[df.keep].reset_index(drop=True)
    caps = df.caption.str.strip().tolist()
    os.makedirs(a.out, exist_ok=True)
    if a.compare:
        res = {}
        for mn in a.compare:
            E = embed(mn, caps)
            st, _ = pair_stats(E)
            res[mn] = st
            print(mn, json.dumps(st), flush=True)
        json.dump(res, open(os.path.join(a.out, "model_compare.json"), "w"), indent=1)
        return
    E = embed(a.model, caps)
    np.save(os.path.join(a.out, "emb.npy"), E)
    df[["audiocap_id"]].to_parquet(os.path.join(a.out, "emb_index.parquet"))
    st, s = pair_stats(E)
    st["model"] = a.model; st["n"] = len(E)
    hist, edges = np.histogram(s, bins=40, range=(-0.2, 1.0))
    st["hist"] = dict(counts=hist.tolist(), edges=edges.tolist())
    json.dump(st, open(os.path.join(a.out, "sim_stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != "hist"}, indent=1))
    # top-K neighbours (excluding self), chunked
    K = a.topk
    nbr = np.zeros((len(E), K), np.int32); nsim = np.zeros((len(E), K), np.float16)
    Et = torch.from_numpy(E)
    for st_ in range(0, len(E), 2048):
        S = Et[st_:st_ + 2048] @ Et.T
        S[torch.arange(S.shape[0]), torch.arange(st_, st_ + S.shape[0])] = -2
        v, ix = torch.topk(S, K, dim=1)
        nbr[st_:st_ + len(ix)] = ix.numpy(); nsim[st_:st_ + len(ix)] = v.numpy()
    np.save(os.path.join(a.out, "nbr_idx.npy"), nbr); np.save(os.path.join(a.out, "nbr_sim.npy"), nsim)
    print("saved", a.out)


if __name__ == "__main__":
    main()
