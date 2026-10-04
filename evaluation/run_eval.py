"""Evaluation driver: tokenizer comparison, bits-per-byte / perplexity, MC benchmarks, samples.

    python -m evaluation.run_eval --config configs/eval/eval_smoke.yaml

Every section is optional (``enabled: false`` or absent). Results are written to
``eval.output_dir`` as ``results.json`` and a human-readable ``report.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.config import parse_config, save_run_metadata, to_container
from common.utils import get_logger, read_jsonl, set_seed, setup_logging, torch_dtype

log = get_logger("evaluation")


def load_text_sets(spec: dict[str, Any]) -> dict[str, list[str]]:
    """``{processed_dir, split, max_docs, max_chars, sets: {name: [categories]}, files: {name: path}}``."""
    processed = Path(spec["processed_dir"])
    split = spec.get("split", "eval")
    max_docs = int(spec.get("max_docs", 200))
    max_chars = int(spec.get("max_chars", 4000))
    out: dict[str, list[str]] = {}
    for name, cats in (spec.get("sets") or {}).items():
        texts: list[str] = []
        for cat in cats:
            for f in sorted((processed / cat).glob(f"{split}-*.jsonl*")):
                texts.extend(r["text"][:max_chars] for r in read_jsonl(f))
        out[name] = texts[:max_docs]
        if not out[name]:
            log.warning("Text set %s is empty (categories %s, split %s)", name, cats, split)
    for name, path in (spec.get("files") or {}).items():
        out[name] = [r["text"][:max_chars] for r in read_jsonl(path)][:max_docs]
    return out


def _load_model(path: str, dtype: str | None, device: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype(dtype))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tok


def run(cfg) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from evaluation.benchmarks import generate_samples, run_task
    from evaluation.perplexity import evaluate_model
    from evaluation.tokenizer_compare import compare, to_markdown

    set_seed(int(cfg.seed))
    e = to_container(cfg.eval)
    out_dir = Path(e["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, out_dir, "kural_eval")
    results: dict[str, Any] = {}
    md: list[str] = [f"# Evaluation report — {cfg.get('run_name', 'kural')}\n"]
    texts = load_text_sets(e["texts"]) if e.get("texts") else {}

    tc = e.get("tokenizers") or {}
    if tc.get("enabled", True) and tc.get("list"):
        toks = {}
        for t in tc["list"]:
            try:
                toks[t["name"]] = AutoTokenizer.from_pretrained(t["path"])
            except Exception as exc:
                log.error("Tokenizer %s (%s) could not be loaded: %s", t["name"], t["path"], exc)
        results["tokenizers"] = compare(toks, texts)
        md += ["## Tokenizers\n", "Fertility = tokens per word (lower is better); % = token count vs baseline.\n",
               to_markdown(results["tokenizers"], tc.get("baseline")), ""]

    models = [m for m in (e.get("models") or []) if m.get("enabled", True)]
    ppl_cfg = e.get("perplexity") or {}
    bench_cfg = [b for b in (e.get("benchmarks") or []) if b.get("enabled", True)]
    gen_cfg = e.get("generation") or {}
    for m in models:
        name = m["name"]
        log.info("Evaluating model %s (%s)", name, m["path"])
        try:
            model, tok = _load_model(m["path"], m.get("dtype"), m.get("device"))
        except Exception as exc:
            log.error("Model %s could not be loaded: %s", name, exc)
            continue
        r: dict[str, Any] = {}
        if ppl_cfg.get("enabled", True) and texts:
            r["lm"] = evaluate_model(model, tok, texts, max_length=int(ppl_cfg.get("max_length", 1024)),
                                     stride=int(ppl_cfg.get("stride", 512)))
        if bench_cfg:
            r["benchmarks"] = {}
            for spec in bench_cfg:
                try:
                    r["benchmarks"][spec["name"]] = run_task(model, tok, spec)
                except Exception as exc:
                    log.error("Benchmark %s failed: %s: %s", spec["name"], type(exc).__name__, exc)
        if gen_cfg.get("enabled", True) and gen_cfg.get("prompts"):
            r["samples"] = generate_samples(model, tok, list(gen_cfg["prompts"]), chat=bool(m.get("chat", False)),
                                            max_new_tokens=int(gen_cfg.get("max_new_tokens", 64)),
                                            **(gen_cfg.get("kwargs") or {"do_sample": False}))
        results.setdefault("models", {})[name] = r
        del model

    if results.get("models"):
        md.append("## Language modelling (bits per byte — lower is better)\n")
        md.append("Comparable across models/tokenizers within a column. Not comparable across columns: "
                  "Tamil script is 3 UTF-8 bytes per character vs 1 for Latin.\n")
        sets = sorted({s for r in results["models"].values() for s in r.get("lm", {})})
        md.append("| model | " + " | ".join(sets) + " |")
        md.append("|---|" + "---:|" * len(sets))
        for name, r in results["models"].items():
            md.append(f"| {name} | " + " | ".join(
                f"{r['lm'][s]['bits_per_byte']:.3f}" if s in r.get("lm", {}) else "–" for s in sets) + " |")
        tasks = sorted({t for r in results["models"].values() for t in r.get("benchmarks", {})})
        if tasks:
            md += ["", "## Benchmarks (acc_norm, zero-shot log-likelihood)\n",
                   "| model | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
            for name, r in results["models"].items():
                b = r.get("benchmarks", {})
                md.append(f"| {name} | " + " | ".join(f"{b[t]['acc_norm']:.3f}" if t in b else "–" for t in tasks) + " |")
        for name, r in results["models"].items():
            if r.get("samples"):
                md += ["", f"### Samples — {name}\n"]
                md += [f"- **{s['prompt']}** → {s['completion'].strip()}" for s in r["samples"]]

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("Wrote %s", out_dir / "report.md")
    return results


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Kural evaluation suite", argv))


if __name__ == "__main__":
    main()
