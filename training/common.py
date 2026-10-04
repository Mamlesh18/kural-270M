"""Trainer plumbing shared by continued pretraining and SFT."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from common.config import to_container
from common.utils import get_logger, resolve_precision
from training.losses import label_only_loss  # noqa: F401  (re-exported)

log = get_logger(__name__)


def setup_wandb(cfg, run_name: str) -> list[str]:
    """Configure W&B through environment variables; returns the ``report_to`` list.

    Credentials are never read from config: W&B picks up ``WANDB_API_KEY`` (or a prior
    ``wandb login``) itself. ``wandb.mode: offline`` logs locally without credentials.
    """
    w = cfg.get("wandb") or {}
    if not w.get("enabled", False) or str(w.get("mode", "online")) == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        return ["none"]
    try:
        import wandb  # noqa: F401
    except ImportError:
        log.warning("wandb not installed; tracking disabled")
        return ["none"]
    os.environ["WANDB_PROJECT"] = str(w.get("project") or "kural-270m")
    if w.get("entity"):
        os.environ["WANDB_ENTITY"] = str(w["entity"])
    os.environ["WANDB_MODE"] = str(w.get("mode", "online"))
    if w.get("group"):
        os.environ["WANDB_RUN_GROUP"] = str(w["group"])
    if w.get("tags"):
        os.environ["WANDB_TAGS"] = ",".join(map(str, w["tags"]))
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ["WANDB_NAME"] = run_name
    return ["wandb"]


def build_training_args(cfg, output_dir: str, report_to: list[str], run_name: str):
    from transformers import TrainingArguments

    t = to_container(cfg.train)
    args = dict(t.get("args") or {})
    args.update(resolve_precision(t.get("precision", "auto")))
    args.setdefault("seed", int(cfg.seed))
    args.setdefault("data_seed", int(cfg.seed))
    args.update(output_dir=output_dir, report_to=report_to, run_name=run_name)
    if args.get("gradient_checkpointing"):
        args.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    if not torch.cuda.is_available():
        if str(args.get("optim", "")).endswith("_fused"):
            args["optim"] = "adamw_torch"  # fused AdamW is CUDA-only
        args["dataloader_pin_memory"] = False
    return TrainingArguments(**args)


def resolve_resume(setting: Any, output_dir: str) -> str | bool | None:
    """``auto`` → latest checkpoint in output_dir if any; ``true``/``false``; or an explicit path."""
    if setting in (None, False, "false", "none"):
        return None
    if setting in ("auto", True, "true"):
        last = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
        if last:
            log.info("Resuming from %s", last)
            return last
        if setting != "auto":
            raise FileNotFoundError(f"resume=true but no checkpoint in {output_dir}")
        return None
    return str(setting)


class KuralTrainer(Trainer):
    """Adds perplexity for every ``*_loss`` eval metric and a running token count.

    ``label_only_logits=True`` switches to :func:`label_only_loss` (for SFT).
    """

    def __init__(self, *args, tokens_per_step: int | None = None, label_only_logits: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_per_step = tokens_per_step
        self.label_only_logits = label_only_logits
        self._extra: dict[str, list[float]] = {}

    def record(self, **metrics: float) -> None:
        """Queue extra training metrics (e.g. DPO reward margins); averaged into the next log line."""
        for k, v in metrics.items():
            self._extra.setdefault(k, []).append(float(v))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.label_only_logits or return_outputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        return label_only_loss(model, inputs["input_ids"], inputs.get("attention_mask"), inputs["labels"],
                               num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not self.label_only_logits:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        # The default evaluation materializes (and keeps) full-vocabulary logits for every token —
        # ~3 GB extra for Gemma's 262k vocab. Only the loss is needed.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = label_only_loss(model, inputs["input_ids"], inputs.get("attention_mask"), inputs["labels"])
        return loss.detach(), None, None

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        for key in list(logs):
            if key.endswith("_loss") and key.startswith("eval"):
                val = logs[key]
                logs[key[:-5] + "_ppl"] = math.exp(val) if val < 50 else float("inf")
        if "loss" in logs and self._extra:
            logs.update({k: sum(v) / len(v) for k, v in self._extra.items()})
            self._extra = {}
        if "loss" in logs and self.tokens_per_step:
            logs["tokens_seen"] = float(self.state.global_step * self.tokens_per_step)
        if self.is_world_process_zero():
            # stderr logging stays live when stdout is redirected (tqdm's writes are buffered)
            log.info("step %d | %s", self.state.global_step, " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()
                if k != "total_flos"))
        super().log(logs, *args, **kwargs)


class SampleGenerationCallback(TrainerCallback):
    """Greedy continuations of fixed prompts after each evaluation (logged + W&B table)."""

    def __init__(self, tokenizer, prompts: list[str], max_new_tokens: int = 48, chat: bool = False):
        self.tok = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.chat = chat
        self._last_step = -1

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # With several eval sets, on_evaluate fires once per set: sample once per step.
        if not state.is_world_process_zero or model is None or not self.prompts or state.global_step == self._last_step:
            return
        self._last_step = state.global_step
        was_training = model.training
        model.eval()
        rows = []
        for prompt in self.prompts:
            if self.chat:
                text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                    add_generation_prompt=True)
                enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
            else:
                enc = self.tok(prompt, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                 pad_token_id=self.tok.pad_token_id)
            completion = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rows.append([state.global_step, prompt, completion])
            log.info("[sample step %d] %s ⟶ %s", state.global_step, prompt, completion.replace("\n", " ⏎ "))
        if was_training:
            model.train()
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"samples": wandb.Table(columns=["step", "prompt", "completion"], data=rows)},
                          commit=False)
        except Exception:
            pass
