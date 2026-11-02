"""Merge <bench>_<setting>.shard*of*.jsonl into <bench>_<setting>.jsonl (last non-error record per qid wins)."""
import glob
import json
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "runs/baselines"
groups = {}
for p in glob.glob(os.path.join(root, "**", "*.shard*of*.jsonl"), recursive=True):
    base = re.sub(r"\.shard\d+of\d+", "", p)
    groups.setdefault(base, []).append(p)
for base, parts in groups.items():
    recs = {}
    for p in ([base] if os.path.exists(base) else []) + sorted(parts):
        for l in open(p):
            j = json.loads(l)
            if j.get("error") is None or j["qid"] not in recs:
                recs[j["qid"]] = j
    with open(base, "w") as f:
        for j in recs.values():
            f.write(json.dumps(j) + "\n")
    print(base, len(recs))
