"""GRPO — Group Relative Policy Optimization (Shao et al. 2024, DeepSeekMath / R1).

Reinforcement learning without a value network or reward model:

1. for each prompt, sample ``G`` completions from the current policy;
2. score them with rule-based rewards (``training/rewards.py``);
3. advantage of each completion = (reward − group mean) / group std;
4. update the policy to raise the log-prob of above-average completions, with a KL
   penalty (k3 estimator) to a frozen reference (the base model — free with LoRA,
   where the reference is the same network with the adapter disabled).

This is the on-policy single-update variant (one optimizer step per batch of
generations, so the importance ratio is 1 and no clipping is needed). Generation
dominates the cost; on a laptop CPU keep ``num_generations``, ``max_new_tokens`` and the
number of steps small.

Outputs in ``output_dir``: ``metrics.jsonl``-compatible events via ``emit``,
``samples.jsonl`` (every scored completion), resumable ``checkpoint-N/`` (adapter +
optimizer + scheduler + step).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from common.utils import get_logger
from training.losses import per_token_logps
from training.rewards import combined

log = get_logger(__name__)


class GRPORunner:
    def __init__(self, model, tok, prompts: list[dict[str, Any]], gcfg: Mapping[str, Any],
                 targs: Mapping[str, Any], output_dir: str | Path, emit: Callable[[dict], None],
                 ref_model=None, seed: int = 0):
        self.model, self.tok, self.prompts = model, tok, prompts
        self.g = dict(gcfg)
        self.t = dict(targs)
        self.out = Path(output_dir)
        self.emit = emit
        self.ref_model = ref_model
        self.rng = random.Random(seed)
        self.num_gen = int(self.g.get("num_generations", 4))
        self.beta = float(self.g.get("beta_kl", 0.04))
        self.weights = dict(self.g.get("rewards") or {"reference_f1": 1.0, "tamil_script": 0.3, "no_repetition": 0.2})
        self.reward_kwargs = dict(self.g.get("reward_kwargs") or {})
        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        self.eos_ids = [tok.eos_token_id] + ([eot] if isinstance(eot, int) and eot != tok.unk_token_id else [])

    # ------------------------------------------------------------------ helpers
    def _prompt_ids(self, messages) -> torch.Tensor:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids[:, -int(self.g.get("max_prompt_tokens", 384)):]

    @torch.no_grad()
    def _generate(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        out = self.model.generate(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), do_sample=True,
            temperature=float(self.g.get("temperature", 0.9)), top_p=float(self.g.get("top_p", 0.95)),
            max_new_tokens=int(self.g.get("max_new_tokens", 64)), num_return_sequences=self.num_gen,
            eos_token_id=self.eos_ids, pad_token_id=self.tok.pad_token_id)
        self.model.train()
        return out

    def _completion_masks(self, seqs: torch.Tensor, prompt_len: int):
        """labels / attention for completions: tokens up to and including the first EOS are trained."""
        labels = torch.full_like(seqs, -100)
        attn = torch.zeros_like(seqs)
        attn[:, :prompt_len] = 1
        texts, lengths = [], []
        for i in range(seqs.shape[0]):
            comp = seqs[i, prompt_len:].tolist()
            end = len(comp)
            for j, t in enumerate(comp):
                if t in self.eos_ids:
                    end = j + 1
                    break
            labels[i, prompt_len:prompt_len + end] = seqs[i, prompt_len:prompt_len + end]
            attn[i, prompt_len:prompt_len + end] = 1
            texts.append(self.tok.decode(comp[:end], skip_special_tokens=True).strip())
            lengths.append(end)
        return labels, attn, texts, lengths

    @torch.no_grad()
    def _ref_logps(self, seqs, attn, labels):
        if self.ref_model is not None:
            return per_token_logps(self.ref_model, seqs, attn, labels)[0]
        with self.model.disable_adapter():
            return per_token_logps(self.model, seqs, attn, labels)[0]

    def _save(self, step: int, opt, sched) -> None:
        ck = self.out / f"checkpoint-{step}"
        ck.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(ck))
        torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, ck / "grpo_state.pt")
        for old in sorted(self.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[:-2]:
            import shutil

            shutil.rmtree(old, ignore_errors=True)

    def _resume(self, opt, sched) -> int:
        cks = sorted(self.out.glob("checkpoint-*/grpo_state.pt"), key=lambda p: int(p.parent.name.split("-")[1]))
        if not cks:
            return 0
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        ck = cks[-1].parent
        set_peft_model_state_dict(self.model, load_file(str(ck / "adapter_model.safetensors")))
        state = torch.load(cks[-1], weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        log.info("Resumed GRPO from %s (step %d)", ck, state["step"])
        return int(state["step"])

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict[str, float]:
        from transformers import get_cosine_schedule_with_warmup

        max_steps = int(self.t.get("max_steps", 20))
        per_step = int(self.t.get("per_device_train_batch_size", 2))  # prompts per optimizer step
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(self.t.get("learning_rate", 5e-6)),
                                weight_decay=float(self.t.get("weight_decay", 0.0)))
        warmup = int(math.ceil(float(self.t.get("warmup_ratio", 0.0)) * max_steps))
        sched = get_cosine_schedule_with_warmup(opt, warmup, max_steps)
        start = self._resume(opt, sched)
        order = list(range(len(self.prompts)))
        self.rng.shuffle(order)
        cursor = start * per_step
        samples_f = (self.out / "samples.jsonl").open("a", encoding="utf-8")
        log_every = int(self.t.get("logging_steps", 1))
        save_every = int(self.t.get("save_steps", 10))
        self.emit({"event": "train_begin", "max_steps": max_steps, "step": start})
        agg: dict[str, list[float]] = {}
        t_step = time.time()
        for step in range(start + 1, max_steps + 1):
            opt.zero_grad(set_to_none=True)
            for _ in range(per_step):
                p = self.prompts[order[cursor % len(order)]]
                cursor += 1
                prompt_ids = self._prompt_ids(p["messages"])
                seqs = self._generate(prompt_ids)
                labels, attn, texts, lengths = self._completion_masks(seqs, prompt_ids.shape[1])
                user_text = p["messages"][-1]["content"]
                scored = [combined(user_text, t, p.get("reference"), self.weights, **self.reward_kwargs) for t in texts]
                rewards = torch.tensor([s[0] for s in scored], dtype=torch.float32)
                std = rewards.std(unbiased=False)
                adv = (rewards - rewards.mean()) / (std + 1e-4)
                for t, (r, parts), ln in zip(texts, scored, lengths):
                    samples_f.write(json.dumps({"step": step, "prompt": user_text, "reference": p.get("reference"),
                                                "completion": t, "reward": r, "parts": parts, "tokens": ln},
                                               ensure_ascii=False) + "\n")
                agg.setdefault("reward_mean", []).append(rewards.mean().item())
                agg.setdefault("reward_std", []).append(std.item())
                agg.setdefault("completion_tokens", []).append(sum(lengths) / len(lengths))
                for name in self.weights:
                    vals = [s[1][name] for s in scored if name in s[1]]
                    if vals:
                        agg.setdefault(f"reward/{name}", []).append(sum(vals) / len(vals))
                if std.item() < 1e-6:
                    agg.setdefault("groups_skipped", []).append(1.0)  # identical rewards → no signal
                    continue
                agg.setdefault("groups_skipped", []).append(0.0)
                pol, keep = per_token_logps(self.model, seqs, attn, labels)
                ref = self._ref_logps(seqs, attn, labels)
                diff = ref - pol
                kl = torch.exp(diff) - diff - 1
                ratio = torch.exp(pol - pol.detach())  # = 1, carries the policy gradient
                tok_obj = ratio * adv[:, None] - self.beta * kl
                seq_obj = (tok_obj * keep).sum(1) / keep.sum(1).clamp(min=1)
                loss = -seq_obj.mean() / per_step
                loss.backward()
                agg.setdefault("loss", []).append(loss.item() * per_step)
                agg.setdefault("kl", []).append(((kl * keep).sum() / keep.sum().clamp(min=1)).item())
            gn = torch.nn.utils.clip_grad_norm_(params, float(self.t.get("max_grad_norm", 1.0)))
            opt.step()
            sched.step()
            agg.setdefault("grad_norm", []).append(float(gn))
            if step % log_every == 0 or step == max_steps:
                m = {k: sum(v) / len(v) for k, v in agg.items() if v}
                m.update(step=step, max_steps=max_steps, learning_rate=sched.get_last_lr()[0],
                         seconds_per_step=(time.time() - t_step) / max(log_every, 1))
                self.emit(m)
                log.info("GRPO step %d/%d | %s", step, max_steps,
                         " ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, float)))
                agg, t_step = {}, time.time()
            samples_f.flush()
            if step % save_every == 0 and step < max_steps:
                self._save(step, opt, sched)
        samples_f.close()
        return {"steps": max_steps}
