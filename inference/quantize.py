"""Post-training quantization + quality check.

    python -m inference.quantize --config configs/inference/quantize.yaml

Methods (``quantize.methods``, run in order, each to its own sub-directory):

* ``dynamic_int8``  PyTorch dynamic int8 on nn.Linear (CPU, no calibration, no extra deps).
                    Embeddings stay fp32 — for Gemma 270M they are most of the weights, so
                    size savings are modest; the speed-up is on the matmuls.
* ``bnb_8bit`` / ``bnb_4bit``  bitsandbytes LLM.int8 / NF4 (CUDA, needs ``bitsandbytes``);
                    saved with ``save_pretrained`` and reloadable by transformers.
* ``gguf``          llama.cpp export: ``convert_hf_to_gguf.py`` (f16) then ``llama-quantize``
                    for each type in ``gguf_types`` (e.g. Q8_0, Q4_K_M). Needs
                    ``LLAMA_CPP_DIR`` (checkout with built binaries). The Gemma 3 converter
                    reads ``tokenizer.model``, which ``tokenizer.adapt`` / ``train_tokenizer``
                    always export alongside the HF tokenizer.

After each torch-loadable method, bits-per-byte on the configured eval texts is
compared with the unquantized model and written to ``quantization_report.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from common.config import parse_config, save_run_metadata, to_container
from common.utils import get_logger, set_seed, setup_logging

log = get_logger("inference.quantize")


def _file_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 2**20
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


def dynamic_int8(model_path: str, out: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).eval()
    try:  # torch.ao.quantization is deprecated in recent torch but still functional
        from torch.ao.quantization import quantize_dynamic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch dynamic quantization unavailable; use bnb_* or gguf") from exc
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel, out / "quantized_model.pt")
    AutoTokenizer.from_pretrained(model_path).save_pretrained(out)
    model.config.save_pretrained(out)
    return qmodel


def bnb(model_path: str, out: Path, bits: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes quantization requires CUDA")
    q = BitsAndBytesConfig(load_in_4bit=bits == 4, load_in_8bit=bits == 8, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=q, device_map="auto")
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(out)
    return model


def gguf(model_path: str, out: Path, types: list[str], llama_cpp_dir: str | None) -> dict[str, float]:
    if not llama_cpp_dir:
        raise RuntimeError("Set LLAMA_CPP_DIR (or quantize.llama_cpp_dir) to a llama.cpp checkout")
    root = Path(llama_cpp_dir)
    convert = root / "convert_hf_to_gguf.py"
    if not (Path(model_path) / "tokenizer.model").exists():
        raise FileNotFoundError("GGUF export needs tokenizer.model next to the HF model")
    out.mkdir(parents=True, exist_ok=True)
    f16 = out / "model-f16.gguf"
    subprocess.run([sys.executable, str(convert), model_path, "--outfile", str(f16), "--outtype", "f16"], check=True)
    sizes = {"f16": _file_mb(f16)}
    binary = next((p for p in (root / "build" / "bin" / "llama-quantize", root / "build" / "bin" / "llama-quantize.exe",
                               root / "build" / "bin" / "Release" / "llama-quantize.exe", root / "llama-quantize")
                   if p.exists()), None)
    if types and binary is None:
        raise FileNotFoundError(f"llama-quantize not found under {root} (build llama.cpp first)")
    for t in types:
        dst = out / f"model-{t}.gguf"
        subprocess.run([str(binary), str(f16), str(dst), t], check=True)
        sizes[t] = _file_mb(dst)
    return sizes


def run(cfg) -> dict[str, Any]:
    from evaluation.perplexity import evaluate_model
    from evaluation.run_eval import load_text_sets

    set_seed(int(cfg.seed))
    q = to_container(cfg.quantize)
    src = q["model_path"]
    out_root = Path(q["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, out_root, "kural_quantize")
    texts = load_text_sets(q["eval_texts"]) if q.get("eval_texts") else {}
    ppl_kw = {"max_length": int(q.get("max_length", 512)), "stride": int(q.get("stride", 256))}
    report: dict[str, Any] = {"source": src, "source_size_mb": _file_mb(Path(src)), "methods": {}}

    if texts:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(src)
        base = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32).eval()
        log.info("Reference (fp32):")
        report["reference"] = evaluate_model(base, tok, texts, **ppl_kw)
        del base
    for method in q.get("methods", ["dynamic_int8"]):
        out = out_root / method
        entry: dict[str, Any] = {}
        try:
            if method == "dynamic_int8":
                model = dynamic_int8(src, out)
            elif method in ("bnb_8bit", "bnb_4bit"):
                model = bnb(src, out, 8 if method == "bnb_8bit" else 4)
            elif method == "gguf":
                entry["files_mb"] = gguf(src, out, list(q.get("gguf_types", ["Q8_0"])),
                                         q.get("llama_cpp_dir") or os.environ.get("LLAMA_CPP_DIR"))
                model = None
            else:
                raise ValueError(f"Unknown method {method!r}")
            entry["size_mb"] = _file_mb(out)
            if model is not None and texts:
                log.info("%s:", method)
                entry["lm"] = evaluate_model(model, tok, texts, **ppl_kw)
                entry["bpb_delta"] = {k: entry["lm"][k]["bits_per_byte"] - report["reference"][k]["bits_per_byte"]
                                      for k in entry["lm"] if k in report.get("reference", {})}
            entry["status"] = "ok"
        except Exception as exc:
            log.error("Quantization %s failed: %s: %s", method, type(exc).__name__, exc)
            entry = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            if out.exists() and not any(out.iterdir()):
                shutil.rmtree(out)
        report["methods"][method] = entry
    (out_root / "quantization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for m, e in report["methods"].items():
        log.info("  %-14s %-6s size=%s MB  Δbpb=%s", m, e["status"], f"{e.get('size_mb', 0):.1f}",
                 {k: round(v, 4) for k, v in e.get("bpb_delta", {}).items()})
    return report


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Quantize a Kural model", argv))


if __name__ == "__main__":
    main()
