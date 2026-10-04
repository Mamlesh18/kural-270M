"""One entry point for every fine-tuning method.

    python -m training.finetune --config configs/finetune/lora_sft_cpu.yaml [key=value ...]

``finetune.method`` selects the recipe:

=================  ===========================================================  ===========
method             what is trained                                               data
=================  ===========================================================  ===========
full               every weight                                                  sft
frozen_embeddings  all weights except the 168M tied embedding (vocab unchanged)  sft
lora / dora        low-rank adapters on attention + MLP (PEFT)                   sft
qlora              LoRA on a 4-bit base (CUDA + bitsandbytes only)               sft
dpo / orpo / simpo preference optimization, via ``finetune.adapter`` weights      preference
grpo               RL with rule-based rewards (GRPO), via adapter weights        prompts
=================  ===========================================================  ===========

Laptop safety (``finetune.resources``): the process caps its CPU threads, lowers its own
priority so the desktop stays responsive, and refuses to start when the estimated
peak RAM exceeds what is free (``max_ram_fraction``). Everything is written to
``finetune.output_dir``: ``metrics.jsonl`` (live progress for the Studio UI),
``status.json``, ``train.log`` (when launched by the Studio), ``adapter/`` and ``final/``.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from common.config import parse_config, save_run_metadata, to_container
from common.utils import get_logger, set_seed, setup_logging

log = get_logger("training.finetune")

SFT_METHODS = ("full", "frozen_embeddings", "lora", "dora", "qlora")
PREF_METHODS = ("dpo", "orpo", "simpo")
RL_METHODS = ("grpo",)
METHODS = SFT_METHODS + PREF_METHODS + RL_METHODS


# --------------------------------------------------------------------------- resources

def apply_resource_limits(res: dict[str, Any]) -> dict[str, Any]:
    import psutil
    import torch

    phys = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    threads = int(res.get("threads") or max(1, phys - 1))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # can only be set once per process
        pass
    prio = "normal"
    if res.get("low_priority", True):
        try:
            p = psutil.Process()
            if os.name == "nt":
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(10)
            prio = "below-normal"
        except Exception as exc:  # pragma: no cover - platform specific
            log.warning("Could not lower priority: %s", exc)
    info = {"threads": threads, "physical_cores": phys, "priority": prio,
            "ram_free_gb": round(psutil.virtual_memory().available / 1e9, 2)}
    log.info("Resource limits: %s", info)
    return info


def estimate_ram_gb(n_params: float, trainable: float, method: str, batch_tokens: int,
                    hidden: int = 640, layers: int = 18, ref_copy: bool = False) -> float:
    """Rough fp32 peak: weights + grads/Adam for trainable params + activations + chunked logits."""
    weights = n_params * 4
    optim = trainable * 12                     # grad + Adam m + v
    acts = batch_tokens * hidden * layers * 40  # activations kept for backward (fp32; MLP is 3.2x wider)
    logits = 256 * 262_144 * 4 * 2             # one chunk of logits + its gradient (chunked loss)
    ref = n_params * 4 if ref_copy else 0
    overhead = 0.4e9                           # python, torch, tokenizer, dataset
    # Calibration (this laptop, LoRA r=16, 2×384 tokens): estimate 2.4 GB, measured peak RSS 1.9 GB.
    return (weights + optim + acts + logits + ref + overhead) / 1e9


# --------------------------------------------------------------------------- metrics

class MetricsWriter:
    def __init__(self, out_dir: Path):
        self.path = out_dir / "metrics.jsonl"
        self.out_dir = out_dir

    def __call__(self, row: dict[str, Any]) -> None:
        import psutil

        vm = psutil.virtual_memory()
        row = {"time": time.time(), "rss_gb": round(psutil.Process().memory_info().rss / 1e9, 2),
               "system_ram_used_pct": vm.percent,
               **{k: (float(v) if hasattr(v, "__float__") and not isinstance(v, (str, bool)) else v)
                  for k, v in row.items()}}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def status(self, state: str, **extra) -> None:
        (self.out_dir / "status.json").write_text(json.dumps({"state": state, "time": time.time(), **extra},
                                                             indent=2, ensure_ascii=False), encoding="utf-8")


def metrics_callback(writer: MetricsWriter):
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            writer({"event": "train_begin", "max_steps": state.max_steps, "step": state.global_step})

        def on_log(self, args, state, control, logs=None, **kw):
            writer({**(logs or {}), "step": state.global_step, "max_steps": state.max_steps})

    return _CB()


# --------------------------------------------------------------------------- model

def load_model(cfg, method: str, adapter: str):
    """Returns (model, tokenizer, ref_model_or_None, info)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.model_factory import apply_trainable, param_report
    from training.peft_utils import apply_adapter, load_quantized_base

    f = to_container(cfg.finetune)
    path = f["model"]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        from tokenizer.train_tokenizer import DEFAULT_CHAT_TEMPLATE

        tok.chat_template = DEFAULT_CHAT_TEMPLATE
    gc = bool(f.get("gradient_checkpointing", False))
    if method == "qlora":
        model = load_quantized_base(path)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa")
    info = param_report(model)
    ref_model = None
    weights = method if method in SFT_METHODS else adapter
    if weights in ("lora", "dora", "qlora"):
        if gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model = apply_adapter(model, weights, f.get("lora"), gradient_checkpointing=gc)
    else:
        apply_trainable(model, "no_embeddings" if weights == "frozen_embeddings" else "all")
        if gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if method == "dpo" or method in RL_METHODS:
            log.warning("%s with full-weight updates keeps a second frozen model copy as reference (+%.1f GB)",
                        method.upper(), info["params_total"] * 4 / 1e9)
            ref_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
    model.config.use_cache = not gc
    info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tok, ref_model, info


# --------------------------------------------------------------------------- run

def run(cfg) -> Path:
    set_seed(int(cfg.seed))
    f = to_container(cfg.finetune)
    method = f["method"]
    if method not in METHODS:
        raise ValueError(f"finetune.method must be one of {METHODS}")
    adapter = f.get("adapter", "lora")
    if method in RL_METHODS and adapter not in ("lora", "dora"):
        raise ValueError("GRPO in this repo trains adapters (finetune.adapter: lora | dora)")
    out = Path(f["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, out, "kural_finetune")
    writer = MetricsWriter(out)
    writer.status("starting", method=method)
    res = apply_resource_limits(f.get("resources") or {})

    try:
        model, tok, ref_model, info = load_model(cfg, method, adapter)
        import psutil

        t = to_container(cfg.train)
        a = t.get("args") or {}
        seq, mult = _sequence_shape(cfg, method)
        est = estimate_ram_gb(info["params_total"], info["trainable_params"], method,
                              int(a.get("per_device_train_batch_size", 1)) * seq * mult, ref_copy=ref_model is not None)
        # Compare with RAM that was free *before* the model was loaded (the estimate includes the weights).
        free = res["ram_free_gb"]
        info.update(res)
        info.update(ram_estimate_gb=round(est, 2), ram_free_gb=round(free, 2))
        writer({"event": "model_loaded", **info})
        limit = float((f.get("resources") or {}).get("max_ram_fraction", 0.9))
        if est > free * limit and not (f.get("resources") or {}).get("ignore_ram_check", False):
            raise MemoryError(f"Estimated peak RAM {est:.1f} GB exceeds {limit:.0%} of free RAM ({free:.1f} GB). "
                              "Use LoRA, a smaller batch / max_seq_len, gradient checkpointing, or close other apps.")

        if method in SFT_METHODS:
            final = _run_sft(cfg, model, tok, method, out, writer)
        elif method in PREF_METHODS:
            final = _run_preference(cfg, model, tok, ref_model, method, adapter, out, writer)
        else:
            final = _run_grpo(cfg, model, tok, ref_model, out, writer)
        (final / "kural_model_info.json").write_text(json.dumps({**info, "method": method, "adapter": adapter,
                                                                 "base_model": f["model"]}, indent=2), encoding="utf-8")
        writer.status("completed", method=method, final_dir=str(final))
        writer({"event": "done", "final_dir": str(final)})
        return final
    except BaseException as exc:
        writer.status("failed" if not isinstance(exc, KeyboardInterrupt) else "stopped",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
        raise


def _sequence_shape(cfg, method: str) -> tuple[int, int]:
    """(max sequence length, sequences per example) used for the RAM estimate."""
    if method in SFT_METHODS:
        return int((cfg.get("sft") or {}).get("max_seq_len", 384)), 1
    if method in PREF_METHODS:
        return int((cfg.get("preference") or {}).get("max_seq_len", 384)), 2   # chosen + rejected
    g = cfg.get("grpo") or {}
    return int(g.get("max_prompt_tokens", 256)) + int(g.get("max_new_tokens", 64)), int(g.get("num_generations", 4))


def _training_args(cfg, out: Path):
    from training.common import build_training_args

    return build_training_args(cfg, str(out), ["none"], str(cfg.run_name))


def _finish(model, tok, trainer, out: Path, method_weights: str, base: str) -> Path:
    from training.peft_utils import save_adapter_and_merged

    if method_weights in ("lora", "dora", "qlora"):
        return save_adapter_and_merged(model, tok, out, base_path=base, quantized_base=method_weights == "qlora")
    final = out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    return final


def _run_sft(cfg, model, tok, method, out, writer) -> Path:
    from training.chat_data import ChatCollator
    from training.common import KuralTrainer, resolve_resume
    from training.sft import build_datasets

    train_ds, eval_ds = build_datasets(cfg, tok)
    writer({"event": "data", "train_examples": len(train_ds), "eval_examples": len(eval_ds)})
    trainer = KuralTrainer(model=model, args=_training_args(cfg, out), train_dataset=train_ds, eval_dataset=eval_ds,
                           data_collator=ChatCollator(tok.pad_token_id), processing_class=tok,
                           callbacks=[metrics_callback(writer)], label_only_logits=True)
    trainer.train(resume_from_checkpoint=resolve_resume(cfg.train.get("resume", "auto"), str(out)))
    writer({**trainer.evaluate(), "event": "final_eval"})
    return _finish(model, tok, trainer, out, method, cfg.finetune.model)


def _run_preference(cfg, model, tok, ref_model, method, adapter, out, writer) -> Path:
    from training.common import resolve_resume
    from training.preference import PairCollator, PreferenceTrainer, RejectedGenerator, build_pair_datasets

    p = to_container(cfg.preference)
    gen = None
    if any(s.get("format") == "self_generated" for s in p["sources"]):
        gen = RejectedGenerator(model, tok, out / "self_generated_cache.jsonl",
                                int(p.get("rejected_max_new_tokens", 96)))
    train_ds, eval_ds = build_pair_datasets(p, tok, int(cfg.seed), gen)
    writer({"event": "data", "train_examples": len(train_ds), "eval_examples": len(eval_ds)})
    trainer = PreferenceTrainer(
        model=model, args=_training_args(cfg, out), train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=PairCollator(tok.pad_token_id), processing_class=tok, callbacks=[metrics_callback(writer)],
        loss_kind=method, beta=float(p.get("beta", 0.1)), orpo_lambda=float(p.get("orpo_lambda", 0.1)),
        simpo_gamma=float(p.get("simpo_gamma", 0.5)), ref_model=ref_model)
    trainer.train(resume_from_checkpoint=resolve_resume(cfg.train.get("resume", "auto"), str(out)))
    writer({**trainer.evaluate(), "event": "final_eval"})
    return _finish(model, tok, trainer, out, adapter, cfg.finetune.model)


def _run_grpo(cfg, model, tok, ref_model, out, writer) -> Path:
    from training.chat_data import iter_conversations
    from training.grpo import GRPORunner
    from training.peft_utils import save_adapter_and_merged

    g = to_container(cfg.grpo)
    prompts = []
    for conv in iter_conversations(g["sources"], None):
        msgs = conv["messages"]
        ref = msgs[-1]["content"] if msgs and msgs[-1]["role"] == "assistant" else None
        prompt = msgs[:-1] if ref is not None else msgs
        if prompt and prompt[-1]["role"] == "user":
            prompts.append({"messages": prompt, "reference": ref})
    if not prompts:
        raise ValueError("GRPO needs prompts (grpo.sources)")
    writer({"event": "data", "train_examples": len(prompts)})
    runner = GRPORunner(model, tok, prompts, g, to_container(cfg.train).get("args") or {}, out, writer,
                        ref_model=ref_model, seed=int(cfg.seed))
    runner.train()
    return save_adapter_and_merged(model, tok, out, base_path=cfg.finetune.model)


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Fine-tune with full / LoRA / DoRA / QLoRA / DPO / ORPO / SimPO / GRPO", argv))


if __name__ == "__main__":
    main()
