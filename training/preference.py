"""Preference optimization: DPO, ORPO and SimPO on (prompt, chosen, rejected) pairs.

Classic RLHF trains a reward model and then optimizes the policy with PPO — four model
copies and text generation inside the training loop. These methods reach the same goal
(prefer the chosen answer over the rejected one) with ordinary supervised passes:

* **DPO** (Rafailov et al. 2023): ``-log σ(β·[(π_c − ref_c) − (π_r − ref_r)])`` where π / ref
  are summed log-probs of the response under the policy and a frozen reference model.
  With a LoRA adapter the reference is the same network with the adapter switched off,
  so it costs no extra memory.
* **ORPO** (Hong et al. 2024): SFT loss on the chosen answer + λ · odds-ratio penalty; no
  reference model at all.
* **SimPO** (Meng et al. 2024): ``-log σ(β·(avg π_c − avg π_r) − γ)`` on length-normalized
  log-probs; no reference model.

Pair sources (``preference.sources``):

* ``format: pairs``    rows ``{prompt, chosen, rejected}`` (field names remappable); ``prompt``
                       may be a string or a list of chat messages.
* ``format: ratings``  the chatbot evaluation UI's ratings JSONL: every blind comparison with
                       a winner (A or B) becomes a pair; ties / both-bad are skipped.
* ``format: self_generated``  any SFT source (alpaca / qa / …): chosen = the dataset answer,
                       rejected = what the *starting* model generates for the prompt
                       (SPIN-style). Generation is slow on CPU; results are cached per run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F

from common.utils import get_logger, read_jsonl, stable_fraction, text_hash
from data.sources import iter_rows
from training.chat_data import IGNORE, encode_conversation, row_to_messages
from training.common import KuralTrainer
from training.losses import sequence_logps

log = get_logger(__name__)
LOSSES = ("dpo", "orpo", "simpo")


# --------------------------------------------------------------------------- pair sources

def _prompt_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


def _pairs_from_rows(src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    f = dict(src.get("fields") or {})
    for row in iter_rows(src):
        p, c, r = row.get(f.get("prompt", "prompt")), row.get(f.get("chosen", "chosen")), row.get(f.get("rejected", "rejected"))
        if p and c and r and c != r:
            yield {"messages": _prompt_messages(p), "chosen": str(c), "rejected": str(r)}


def _pairs_from_ratings(src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    path = Path(src["path"])
    if not path.exists():
        log.warning("Ratings file %s does not exist yet — vote in the chatbot UI first", path)
        return
    for r in read_jsonl(path):
        if r.get("mode") != "compare" or r.get("winner") not in ("a", "b"):
            continue
        win, lose = ("a", "b") if r["winner"] == "a" else ("b", "a")
        c, rj = r.get(f"response_{win}"), r.get(f"response_{lose}")
        if r.get("prompt") and c and rj and c.strip() != rj.strip():
            yield {"messages": _prompt_messages(r["prompt"]), "chosen": c.strip(), "rejected": rj.strip()}


def iter_pairs(sources: list[Mapping[str, Any]], generator=None) -> Iterator[dict[str, Any]]:
    for src in sources:
        if not src.get("enabled", True):
            continue
        fmt = src.get("format", "pairs")
        n = 0
        try:
            if fmt == "pairs":
                it = _pairs_from_rows(src)
            elif fmt == "ratings":
                it = _pairs_from_ratings(src)
            elif fmt == "self_generated":
                if generator is None:
                    raise ValueError("self_generated pairs need a generator")
                it = generator.pairs(src)
            else:
                raise ValueError(f"Unknown preference format {fmt!r}")
            for pair in it:
                n += 1
                yield {**pair, "source": src["name"]}
        except Exception as exc:
            if src.get("required", False):
                raise
            log.error("Preference source %s failed and was skipped: %s: %s", src["name"], type(exc).__name__, exc)
        log.info("Preference source %s: %d pairs", src["name"], n)


class RejectedGenerator:
    """Builds SPIN-style pairs: dataset answer (chosen) vs the starting model's own answer (rejected)."""

    def __init__(self, model, tok, cache_file: Path, max_new_tokens: int = 128):
        self.model, self.tok, self.cache_file, self.max_new_tokens = model, tok, Path(cache_file), max_new_tokens
        self.cache: dict[str, str] = {}
        if self.cache_file.exists():
            for row in read_jsonl(self.cache_file):
                self.cache[row["key"]] = row["text"]

    @torch.no_grad()
    def _generate(self, messages: list[dict[str, str]]) -> str:
        key = text_hash(json.dumps(messages, ensure_ascii=False))
        if key in self.cache:
            return self.cache[key]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        eos = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<end_of_turn>")]
        was_training = self.model.training
        self.model.eval()
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, eos_token_id=eos,
                                  pad_token_id=self.tok.pad_token_id)
        if was_training:
            self.model.train()
        reply = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        self.cache[key] = reply
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "text": reply}, ensure_ascii=False) + "\n")
        return reply

    def pairs(self, src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        fmt, fields = src.get("sft_format", "alpaca"), dict(src.get("fields") or {})
        for i, row in enumerate(iter_rows(src)):
            msgs = row_to_messages(row, fmt, fields, src)
            if not msgs or msgs[-1]["role"] != "assistant":
                continue
            prompt, chosen = msgs[:-1], msgs[-1]["content"]
            rejected = self._generate(prompt)
            if i % 10 == 0:
                log.info("  generated %d rejected answers for %s", i + 1, src["name"])
            if rejected and rejected.strip() != chosen.strip():
                yield {"messages": prompt, "chosen": chosen, "rejected": rejected}


# --------------------------------------------------------------------------- encoding

def encode_pair(tok, pair: Mapping[str, Any], max_len: int) -> dict[str, list[int]] | None:
    c = encode_conversation(tok, [*pair["messages"], {"role": "assistant", "content": pair["chosen"]}], max_len)
    r = encode_conversation(tok, [*pair["messages"], {"role": "assistant", "content": pair["rejected"]}], max_len)
    if c is None or r is None:
        return None
    return {"chosen_input_ids": c["input_ids"], "chosen_labels": c["labels"],
            "rejected_input_ids": r["input_ids"], "rejected_labels": r["labels"]}


def build_pair_datasets(pcfg: Mapping[str, Any], tok, seed: int, generator=None):
    from datasets import Dataset

    max_len = int(pcfg.get("max_seq_len", 512))
    eval_fraction = float(pcfg.get("eval_fraction", 0.05))
    train, evals, seen, dropped = [], [], set(), 0
    for pair in iter_pairs(list(pcfg["sources"]), generator):
        key = text_hash(json.dumps([pair["messages"], pair["chosen"], pair["rejected"]], ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        ex = encode_pair(tok, pair, max_len)
        if ex is None:
            dropped += 1
            continue
        (evals if stable_fraction(key, str(seed)) < eval_fraction else train).append(ex)
    if not train:
        raise ValueError("No preference pairs. Add pairs, vote in the chatbot UI, or use a self_generated source.")
    if not evals:
        evals = train[: max(1, len(train) // 10)]
    log.info("Preference pairs: train=%d eval=%d dropped=%d (too long)", len(train), len(evals), dropped)
    return Dataset.from_list(train), Dataset.from_list(evals)


@dataclass
class PairCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        seqs = [(f["chosen_input_ids"], f["chosen_labels"]) for f in features] + \
               [(f["rejected_input_ids"], f["rejected_labels"]) for f in features]
        n = max(len(s[0]) for s in seqs)
        n = ((n + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        ids = torch.full((len(seqs), n), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(seqs), n), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, (x, y) in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x)
            labels[i, :len(y)] = torch.tensor(y)
            mask[i, :len(x)] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# --------------------------------------------------------------------------- losses

def preference_loss(kind: str, pol_c, pol_r, n_c, n_r, ref_c=None, ref_r=None, beta: float = 0.1,
                    orpo_lambda: float = 0.1, simpo_gamma: float = 0.5):
    """Returns ``(loss, metrics)``. ``pol_*``/``ref_*`` are summed response log-probs (B,), ``n_*`` token counts."""
    if kind == "dpo":
        if ref_c is None:
            raise ValueError("DPO needs reference log-probs")
        margin = (pol_c - ref_c) - (pol_r - ref_r)
        loss = -F.logsigmoid(beta * margin).mean()
        return loss, {"reward_chosen": (beta * (pol_c - ref_c)).mean().item(),
                      "reward_rejected": (beta * (pol_r - ref_r)).mean().item(),
                      "reward_accuracy": (margin > 0).float().mean().item()}
    avg_c, avg_r = pol_c / n_c.clamp(min=1), pol_r / n_r.clamp(min=1)
    if kind == "orpo":
        # log odds(p) = log p − log(1 − p) with p = exp(avg log-prob); clamp keeps log1p finite.
        lc, lr = avg_c.clamp(max=-1e-6), avg_r.clamp(max=-1e-6)
        log_odds = (lc - lr) - (torch.log1p(-torch.exp(lc)) - torch.log1p(-torch.exp(lr)))
        nll = -pol_c.sum() / n_c.sum().clamp(min=1)
        ratio = -F.logsigmoid(log_odds).mean()
        return nll + orpo_lambda * ratio, {"sft_nll": nll.item(), "odds_ratio_loss": ratio.item(),
                                           "reward_accuracy": (log_odds > 0).float().mean().item()}
    if kind == "simpo":
        margin = beta * (avg_c - avg_r) - simpo_gamma
        return -F.logsigmoid(margin).mean(), {"reward_margin": (avg_c - avg_r).mean().item(),
                                              "reward_accuracy": (avg_c > avg_r).float().mean().item()}
    raise ValueError(f"Unknown preference loss {kind!r}")


class PreferenceTrainer(KuralTrainer):
    """HF Trainer for DPO / ORPO / SimPO with label-only, chunked log-probs."""

    def __init__(self, *args, loss_kind: str = "dpo", beta: float = 0.1, orpo_lambda: float = 0.1,
                 simpo_gamma: float = 0.5, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_kind not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        self.loss_kind, self.beta, self.orpo_lambda, self.simpo_gamma = loss_kind, beta, orpo_lambda, simpo_gamma
        self.ref_model = ref_model
        # Our loss is a per-pair mean: let the Trainer divide by gradient-accumulation steps.
        self.model_accepts_loss_kwargs = False

    @torch.no_grad()
    def _ref_logps(self, model, inputs):
        if self.ref_model is not None:
            return sequence_logps(self.ref_model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])[0]
        peft_model = model.module if hasattr(model, "module") else model
        if hasattr(peft_model, "disable_adapter"):
            with peft_model.disable_adapter():
                return sequence_logps(peft_model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])[0]
        raise RuntimeError("DPO needs a reference: use a LoRA/DoRA adapter or pass ref_model")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        b = inputs["input_ids"].shape[0] // 2
        pol, n = sequence_logps(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        ref_c = ref_r = None
        if self.loss_kind == "dpo":
            ref = self._ref_logps(model, inputs)
            ref_c, ref_r = ref[:b], ref[b:]
        loss, metrics = preference_loss(self.loss_kind, pol[:b], pol[b:], n[:b], n[b:], ref_c, ref_r,
                                        self.beta, self.orpo_lambda, self.simpo_gamma)
        if model.training:
            self.record(**metrics)
        return (loss, {"logits": pol}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None
