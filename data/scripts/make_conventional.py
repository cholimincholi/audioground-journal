#!/usr/bin/env python
"""E5 'conventional task-specific outputs': copy variants/default rows (same ids, splits, order, audio, layout,
gt_intervals) and rewrite only the response text:
  grounding: unchanged interval answer | duration: duration only | ordering: precedence only | frequency: count only."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import norm_caption
R = "data/variants"
import sys
SRC, DST = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("default", "conventional")   # e.g. lead conventional_lead
DUR = ["It lasts about {d} seconds.", "The sound lasts about {d} seconds.", "Its duration is about {d} seconds."]
ORD = ['"{f}" happens before "{s}".', '"{f}" happens first.', 'The first event is "{f}", followed by "{s}".']
FRQ = ["It occurs {k} times.", "The sound is heard {k} times.", "It happens {k} times in total."]


def pick(lst, rid):
    return lst[int(hashlib.md5(rid.encode()).hexdigest(), 16) % len(lst)]


def convert(r):
    m, t = r["meta"], r["task"]
    if t == "duration":
        s, e = m["gt_exact"][0]
        r["response"] = pick(DUR, m["id"]).format(d=f"{round(e - s, 1):.1f}")
    elif t == "ordering":
        tg = {s["tgt"]: norm_caption(s["caption"]) for s in m["segments"] if s["role"] == "target"}
        mention = [1, 0] if m["mention_swap"] else [0, 1]                 # gt_exact is in mention order
        qa, qb = tg[mention[0]], tg[mention[1]]
        (a0, _), (b0, _) = m["gt_exact"]
        f, s = (qa, qb) if a0 <= b0 else (qb, qa)
        assert f'"{qa}"' in r["instruction"] and f'"{qb}"' in r["instruction"], r["instruction"]
        r["response"] = pick(ORD, m["id"]).format(f=f, s=s)
        m["first_event"] = "first_mentioned" if f == qa else "second_mentioned"
    elif t == "frequency":
        r["response"] = pick(FRQ, m["id"]).format(k=len(r["gt_intervals"]))
    m["response_style"] = "conventional"
    return r


os.makedirs(f"{R}/{DST}", exist_ok=True)
for split in ("train", "val"):
    with open(f"{R}/{SRC}/{split}.jsonl") as fi, open(f"{R}/{DST}/{split}.jsonl", "w") as fo:
        for l in fi:
            fo.write(json.dumps(convert(json.loads(l)), ensure_ascii=False) + "\n")
