"""Human evaluation records (from the chat evaluation UI) and their summary.

Ratings are appended to a JSONL file, one record per judgement:

* ``mode: "compare"`` – blind pairwise vote between ``model_a`` and ``model_b``
  (``winner``: ``a`` | ``b`` | ``tie`` | ``both_bad``), optional per-response scores.
* ``mode: "chat"``    – scores for a single response in free chat.

Scores are 1–5 per criterion (``CRITERIA``). The summary reports, per model,
pairwise win rate (ties count half; "both bad" counts as a tie for both but is
tracked separately) and mean criterion scores, overall and per prompt category.

    python -m evaluation.human_eval --ratings workspace/reports/human_eval/ratings.jsonl
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CRITERIA = {
    "correctness": "Correct / factual",
    "fluency": "Natural Tamil (grammar, register)",
    "helpfulness": "Helpful / follows the request",
}
WINNERS = ("a", "b", "tie", "both_bad")

_lock = threading.Lock()


def append_rating(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    mode = record.get("mode")
    if mode not in ("compare", "chat"):
        raise ValueError("mode must be 'compare' or 'chat'")
    if mode == "compare":
        if record.get("winner") not in WINNERS:
            raise ValueError(f"winner must be one of {WINNERS}")
        if not record.get("model_a") or not record.get("model_b"):
            raise ValueError("compare ratings need model_a and model_b")
    elif not record.get("model"):
        raise ValueError("chat ratings need model")
    for scores in _iter_scores(record):
        for k, v in scores.items():
            if k not in CRITERIA or not (isinstance(v, int) and 1 <= v <= 5):
                raise ValueError(f"invalid score {k}={v!r} (criteria {list(CRITERIA)}, 1-5)")
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _iter_scores(r: dict[str, Any]):
    if r.get("mode") == "compare":
        for side in ("a", "b"):
            s = (r.get("scores") or {}).get(side)
            if s:
                yield s
    elif r.get("scores"):
        yield r["scores"]


def load_ratings(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _new_stats():
    return {"comparisons": 0, "wins": 0, "losses": 0, "ties": 0, "both_bad": 0,
            "score_sum": defaultdict(float), "score_n": defaultdict(int)}


def _finish(s: dict[str, Any]) -> dict[str, Any]:
    n = s["comparisons"]
    return {
        "comparisons": n,
        "wins": s["wins"], "losses": s["losses"], "ties": s["ties"], "both_bad": s["both_bad"],
        "win_rate": (s["wins"] + 0.5 * (s["ties"] + s["both_bad"])) / n if n else None,
        "scores": {k: round(s["score_sum"][k] / s["score_n"][k], 2) for k in CRITERIA if s["score_n"][k]},
        "scored_responses": max(s["score_n"].values(), default=0),
    }


def summarize(ratings: list[dict[str, Any]]) -> dict[str, Any]:
    overall: dict[str, dict] = defaultdict(_new_stats)
    by_cat: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(_new_stats))
    head_to_head: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])  # a-wins, b-wins, ties

    def add_scores(model: str, cat: str, scores: dict[str, int] | None):
        for k, v in (scores or {}).items():
            for s in (overall[model], by_cat[cat][model]):
                s["score_sum"][k] += v
                s["score_n"][k] += 1

    for r in ratings:
        cat = r.get("category") or "custom"
        if r.get("mode") == "compare":
            a, b, w = r["model_a"], r["model_b"], r.get("winner")
            for s in (overall, by_cat[cat]):
                s[a]["comparisons"] += 1
                s[b]["comparisons"] += 1
                if w == "a":
                    s[a]["wins"] += 1
                    s[b]["losses"] += 1
                elif w == "b":
                    s[b]["wins"] += 1
                    s[a]["losses"] += 1
                elif w == "tie":
                    s[a]["ties"] += 1
                    s[b]["ties"] += 1
                else:
                    s[a]["both_bad"] += 1
                    s[b]["both_bad"] += 1
            x, y = sorted((a, b))
            h = head_to_head[(x, y)]
            if w in ("a", "b"):
                winner = a if w == "a" else b
                h[0 if winner == x else 1] += 1
            else:
                h[2] += 1
            add_scores(a, cat, (r.get("scores") or {}).get("a"))
            add_scores(b, cat, (r.get("scores") or {}).get("b"))
        elif r.get("mode") == "chat":
            add_scores(r["model"], cat, r.get("scores"))
            overall[r["model"]]  # make sure the model appears even with no comparisons
            by_cat[cat][r["model"]]

    models = {m: _finish(s) for m, s in overall.items()}
    ranking = sorted(models, key=lambda m: (models[m]["win_rate"] is None, -(models[m]["win_rate"] or 0)))
    return {
        "n_ratings": len(ratings),
        "n_compare": sum(r.get("mode") == "compare" for r in ratings),
        "n_chat": sum(r.get("mode") == "chat" for r in ratings),
        "criteria": CRITERIA,
        "ranking": ranking,
        "models": models,
        "by_category": {c: {m: _finish(s) for m, s in d.items()} for c, d in sorted(by_cat.items())},
        "head_to_head": [{"model_x": x, "model_y": y, "x_wins": v[0], "y_wins": v[1], "ties": v[2]}
                         for (x, y), v in sorted(head_to_head.items())],
    }


def to_markdown(summary: dict[str, Any]) -> str:
    crit = list(summary["criteria"])
    lines = [f"# Human evaluation ({summary['n_compare']} blind comparisons, {summary['n_chat']} chat ratings)", "",
             "| model | comparisons | win rate | W / T / L / both bad | " + " | ".join(crit) + " |",
             "|---|---:|---:|---|" + "---:|" * len(crit)]
    for m in summary["ranking"]:
        s = summary["models"][m]
        wr = f"{100 * s['win_rate']:.0f}%" if s["win_rate"] is not None else "–"
        lines.append(f"| {m} | {s['comparisons']} | {wr} | {s['wins']} / {s['ties']} / {s['losses']} / {s['both_bad']} | "
                     + " | ".join(str(s["scores"].get(k, "–")) for k in crit) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize human evaluation ratings")
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--out", help="write a markdown report here")
    args = ap.parse_args()
    summary = summarize(load_ratings(args.ratings))
    md = to_markdown(summary)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
