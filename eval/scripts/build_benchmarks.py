"""Build unified original/clipped annotation files (+ clipped wavs) for the 3 benchmarks.

usage: python scripts/build_benchmarks.py [--eps 0] [--anchor span] [--no-audio]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agl_eval import BENCH_ROOT  # noqa: E402
from agl_eval.clipping import clip_query  # noqa: E402

SRC = {
    "clotho_moment": ("clotho_moment_test_release.jsonl", lambda v: v.replace(".", "")),
    "unav100": ("unav100-subset_test_release.jsonl", lambda v: v),
    "tut2017": ("tut2017_test_release.jsonl", lambda v: v),
}


def _write_clip(job):
    src, dst, s, e = job
    if os.path.exists(dst):
        return
    info = sf.info(src)
    sr = info.samplerate
    a, b = int(round(s * sr)), min(int(round(e * sr)), info.frames)
    x, _ = sf.read(src, start=a, stop=b, dtype="int16", always_2d=False)
    tmp = dst + ".tmp.wav"
    sf.write(tmp, x, sr, subtype="PCM_16")
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--anchor", default="first")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for bench, (fn, vmap) in SRC.items():
        root = os.path.join(BENCH_ROOT, bench)
        ann = [json.loads(l) for l in open(os.path.join(root, fn))]
        dur_cache = {}
        orig, clipped, jobs = [], [], []
        os.makedirs(os.path.join(root, "audio_clipped" + args.suffix), exist_ok=True)
        for a in ann:
            wav = os.path.join(root, "audio_original", vmap(a["vid"]) + ".wav")
            if wav not in dur_cache:
                dur_cache[wav] = sf.info(wav).duration
            L = dur_cache[wav]
            gt = [list(map(float, w)) for w in a["relevant_windows"]]
            base = dict(qid=str(a["qid"]), bench=bench, query=a["query"], vid=a["vid"])
            orig.append(dict(base, setting="original", audio=wav, L=L, gt=gt, ann_duration=a["duration"],
                             clip_start=0.0, clip_branch="original"))
            c = clip_query(gt, L, eps=args.eps, anchor=args.anchor)
            if c["branch"] == "unchanged":
                cwav = wav
            else:
                cwav = os.path.join(root, "audio_clipped" + args.suffix, f"q{a['qid']}_{vmap(a['vid'])}.wav")
                jobs.append((wav, cwav, c["start"], c["end"]))
            clipped.append(dict(base, setting="clipped", audio=cwav, L=c["L_new"], gt=c["windows"],
                                gt_original=gt, L_original=L, clip_start=c["start"], clip_branch=c["branch"]))
        for name, recs in (("original", orig), ("clipped" + args.suffix, clipped)):
            with open(os.path.join(root, f"{name}.jsonl"), "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        if not args.no_audio:
            with Pool(args.workers) as p:
                p.map(_write_clip, jobs, chunksize=16)
        from collections import Counter
        print(bench, "n=", len(orig), "branches", Counter(r["clip_branch"] for r in clipped),
              "empty_gt_after_clip", sum(1 for r in clipped if not r["gt"]),
              "n_gt_dropped", sum(len(r["gt_original"]) - len(r["gt"]) for r in clipped))


if __name__ == "__main__":
    main()
