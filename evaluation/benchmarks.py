"""Log-likelihood multiple-choice evaluation (lm-eval-harness style, zero-shot).

Each item is ``(context, [choice_0, ...], label)``. A choice's score is the summed
log-probability of its tokens given the context; we report

* ``acc``       argmax of summed log-prob
* ``acc_norm``  argmax of log-prob per UTF-8 byte of the choice (length-robust and
                tokenizer-independent — preferred when comparing tokenizers)

Task adapters turn dataset rows into items:

* ``cloze``      {context, choices, label}               (built-in Tamil sanity set)
* ``copa``       IndicCOPA: premise / choice1 / choice2 / question / label
* ``sentiment``  IndicSentiment: INDIC REVIEW / LABEL (Positive|Negative)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F

from common.utils import get_logger
from data.sources import iter_rows

log = get_logger(__name__)


def _cloze(row, spec):
    return row[spec.get("context_field", "context")], list(row[spec.get("choices_field", "choices")]), \
        int(row[spec.get("label_field", "label")]), row.get("category")


def _copa(row, spec):
    premise = row["premise"].rstrip("।. ")
    connective = spec.get("cause_word", " ஏனென்றால்") if row["question"] == "cause" else spec.get("effect_word", " எனவே")
    return premise + connective, [" " + row["choice1"], " " + row["choice2"]], int(row["label"]), row["question"]


def _sentiment(row, spec):
    review = row[spec.get("text_field", "INDIC REVIEW")]
    prompt = spec.get("prompt", "{text}\nஇந்த விமர்சனம்")
    labels = spec.get("labels", {"Positive": " நேர்மறையானது", "Negative": " எதிர்மறையானது"})
    names = list(labels)
    if row["LABEL"] not in labels:
        return None
    return prompt.format(text=review), [labels[n] for n in names], names.index(row["LABEL"]), None


ADAPTERS = {"cloze": _cloze, "copa": _copa, "sentiment": _sentiment}


def iter_items(spec: Mapping[str, Any]) -> Iterator[tuple[str, list[str], int, str | None]]:
    adapter = ADAPTERS[spec.get("adapter", "cloze")]
    for row in iter_rows(spec):
        item = adapter(row, spec)
        if item is not None:
            yield item


@torch.no_grad()
def choice_logprobs(model, tok, context: str, choices: list[str], device) -> list[float]:
    ctx = [tok.bos_token_id] + tok(context, add_special_tokens=False)["input_ids"]
    seqs, conts = [], []
    for ch in choices:
        cont = tok(ch, add_special_tokens=False)["input_ids"]
        seqs.append(ctx + cont)
        conts.append(len(cont))
    n = max(map(len, seqs))
    pad = tok.pad_token_id
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    out = []
    for i, s in enumerate(seqs):
        start = len(s) - conts[i]
        tgt = torch.tensor(s[start:], device=device)
        out.append(float(logp[i, start - 1:len(s) - 1].gather(-1, tgt[:, None]).sum()))
    return out


def run_task(model, tok, spec: Mapping[str, Any]) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    n = acc = acc_norm = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    limit = spec.get("limit")
    for context, choices, label, cat in iter_items(spec):
        lps = choice_logprobs(model, tok, context, choices, device)
        norm = [lp / max(len(c.encode("utf-8")), 1) for lp, c in zip(lps, choices)]
        hit = int(max(range(len(lps)), key=lps.__getitem__) == label)
        hit_norm = int(max(range(len(norm)), key=norm.__getitem__) == label)
        n += 1
        acc += hit
        acc_norm += hit_norm
        if cat:
            by_cat[cat][0] += hit_norm
            by_cat[cat][1] += 1
        if limit and n >= int(limit):
            break
    res: dict[str, Any] = {"n": n, "acc": acc / max(n, 1), "acc_norm": acc_norm / max(n, 1)}
    if by_cat:
        res["acc_norm_by_category"] = {k: v[0] / v[1] for k, v in sorted(by_cat.items())}
    log.info("  task %-22s n=%4d acc=%.3f acc_norm=%.3f", spec["name"], n, res["acc"], res["acc_norm"])
    return res


@torch.no_grad()
def generate_samples(model, tok, prompts: list[str], chat: bool = False, max_new_tokens: int = 64,
                     **gen_kwargs) -> list[dict[str, str]]:
    device = next(model.parameters()).device
    model.eval()
    out = []
    for p in prompts:
        if chat:
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        else:
            enc = tok(p, return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **gen_kwargs)
        out.append({"prompt": p, "completion": tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)})
    return out
