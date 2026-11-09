"""Re-score saved answers after a scorer fix. Does NOT call the server or the agent.

Only count rows are re-scored, because the fix touches only count parsing. Every other verdict
is carried over unchanged. Writes new files, so the original v1 results are kept.

Usage: python eval/rescore.py --in eval/results_v1.jsonl --cases eval/cases.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_eval import build_report  # noqa: E402
from eval.score import score_row  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="eval/results_v1.jsonl")
    ap.add_argument("--cases", default="eval/cases.jsonl")
    args = ap.parse_args()

    cases = {}
    with open(args.cases, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cases[row["id"]] = row

    src = Path(args.src)
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = []
    for r in rows:
        if r["type"] != "count":
            continue
        response = {"http_status": r["http_status"],
                    "body": {"status": r["status"], "answer": r["answer"]}}
        new = score_row(cases[r["id"]], response, id_to_name={})
        if new["verdict"] != r["verdict"]:
            changed.append((r["id"], r["verdict"], new["verdict"]))
        r["verdict"], r["reason"] = new["verdict"], new["reason"]

    out_jsonl = src.with_name(src.stem + "_rescored.jsonl")
    out_md = src.with_name("report_v1_rescored.md")
    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out_md.write_text(build_report(rows), encoding="utf-8")

    print(f"wrote {out_jsonl} and {out_md}")
    print("changed verdicts:" if changed else "no verdicts changed")
    for rid, before, after in changed:
        print(f"  {rid}: {before} -> {after}")


if __name__ == "__main__":
    main()
