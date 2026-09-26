"""Parameter-efficient fine-tuning: LoRA, DoRA and QLoRA adapters (via 🤗 PEFT).

* **LoRA** freezes all base weights and learns low-rank updates ``W + (α/r)·B·A`` on the
  chosen linear layers — ~1–4M trainable parameters instead of 268M, so gradients and
  Adam state shrink from ~1.6 GB to a few MB. Forward/backward through the frozen
  network is still needed, so on CPU the speed-up is modest; the RAM saving is large.
* **DoRA** (``use_dora``) additionally learns a per-column magnitude; often a bit better
  than LoRA at the same rank, ~20–30% slower.
* **QLoRA** loads the frozen base in 4-bit NF4 (bitsandbytes) and trains LoRA on top.
  bitsandbytes 4-bit kernels need a CUDA GPU, so this path is refused on CPU with a
  clear message rather than silently running something else.

After training, the adapter is saved on its own (small, shareable) *and* merged into a
full model (``final/``) so every other tool in the repo — evaluation, quantization,
the chat UI — can load it like any checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from common.utils import get_logger

log = get_logger(__name__)

# Attention + MLP projections of Gemma 3 (also valid for Llama/Qwen/Mistral-style models).
DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ADAPTER_METHODS = ("lora", "dora", "qlora")


def qlora_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "QLoRA needs a CUDA GPU: bitsandbytes' 4-bit kernels do not run on CPU."
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False, "QLoRA needs `pip install bitsandbytes`."
    return True, ""


def load_quantized_base(path: str, revision: str | None = None):
    """Load a causal LM in 4-bit NF4 for QLoRA (CUDA only)."""
    ok, why = qlora_available()
    if not ok:
        raise RuntimeError(why)
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(path, revision=revision, quantization_config=q, device_map="auto")


def apply_adapter(model, method: str, cfg: Mapping[str, Any] | None = None, gradient_checkpointing: bool = False):
    """Wrap ``model`` with a LoRA / DoRA / QLoRA adapter. Returns the PEFT model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if method not in ADAPTER_METHODS:
        raise ValueError(f"adapter method must be one of {ADAPTER_METHODS}, got {method!r}")
    cfg = dict(cfg or {})
    if method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    lcfg = LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("alpha", 32)),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        target_modules=list(cfg.get("target_modules") or DEFAULT_TARGETS),
        use_dora=(method == "dora"),
        use_rslora=bool(cfg.get("use_rslora", False)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    trainable, total = model.get_nb_trainable_parameters()
    log.info("%s adapter: r=%d alpha=%d targets=%s → %.2fM trainable / %.1fM total (%.2f%%)",
             method.upper(), lcfg.r, lcfg.lora_alpha, sorted(lcfg.target_modules), trainable / 1e6, total / 1e6,
             100 * trainable / total)
    return model


def trainable_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_adapter_and_merged(model, tok, out_dir: str | Path, base_path: str | None = None,
                            quantized_base: bool = False) -> Path:
    """Save ``adapter/`` (PEFT weights only) and ``final/`` (adapter merged into a full fp32 model)."""
    out = Path(out_dir)
    adapter_dir, final_dir = out / "adapter", out / "final"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    if quantized_base:
        # Merging into 4-bit weights would bake quantization error in; reload the base in full precision.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32)
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    else:
        merged = model.merge_and_unload()
    merged.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    log.info("Saved adapter → %s and merged model → %s", adapter_dir, final_dir)
    return final_dir
