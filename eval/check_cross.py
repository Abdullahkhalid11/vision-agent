"""Check cross-image expected sets against the per-image count and yes/no rows.
Usage: python eval/check_cross.py eval/cases.jsonl"""
import json
import sys
from collections import defaultdict

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]

# Positive or negative evidence per (label, image), from count rows and yes/no rows.
evidence = defaultdict(set)
for r in rows:
    if r["type"] == "count" and isinstance(r["expected"], int):
        evidence[(r["label"], r["image"])].add(r["expected"] > 0)
    elif r["type"] in ("exists", "ambiguous") and r.get("label") and r["expected"] in ("yes", "no"):
        evidence[(r["label"], r["image"])].add(r["expected"] == "yes")

ok = True
for r in rows:
    if r["type"] != "cross":
        continue
    label = r["label"]
    derived = sorted(img for (lab, img), vals in evidence.items() if lab == label and True in vals)
    conflicts = [img for (lab, img), vals in evidence.items() if lab == label and len(vals) > 1]
    stated = r["expected"]
    print(f"{r['id']} label={label!r}")
    print(f"   derived from per-image rows: {derived}")
    print(f"   stated in cases.jsonl:       {stated}")
    if conflicts:
        print(f"   CONFLICT in per-image rows: {conflicts}")
        ok = False
    if stated is None:
        print("   MISSING: expected is null")
        ok = False
    elif sorted(stated) != derived:
        print("   MISMATCH")
        ok = False
    else:
        print("   consistent")

print("\nRESULT:", "all cross rows consistent" if ok else "fix the rows above before committing")