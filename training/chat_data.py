"""Instruction / chat data → Gemma chat format with assistant-only loss masking.

Supported row formats (``format`` on each SFT source):

* ``alpaca``           {instruction, input, output}
* ``prompt_response``  {prompt, response}            (e.g. Aya: inputs / targets)
* ``messages``         [{role, content}, ...]        (OpenAI style)
* ``sharegpt``         [{from, value}, ...]
* ``qa``               {context, question, answer}  (reading comprehension, e.g. Tamil SQuAD 2.0).
                       The prompt is ``qa_template``; rows with no answer (SQuAD 2.0
                       "unanswerable") get ``unanswerable_response``, or are dropped if it is null.

Field names are remapped with ``fields: {canonical_name: dataset_column}``.

Rendering matches the tokenizer's chat template::

    <bos><start_of_turn>user\\n{user}<end_of_turn>\\n<start_of_turn>model\\n{assistant}<end_of_turn>\\n

Only assistant content plus its ``<end_of_turn>`` (so the model learns to stop)
contributes to the loss; everything else is labelled -100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch

from common.utils import get_logger
from data.normalize import TextNormalizer
from data.sources import iter_rows, strip_prefixes

log = get_logger(__name__)
IGNORE = -100

# "Read the passage below and answer the question. Passage: ... Question: ..."
DEFAULT_QA_TEMPLATE = "கீழே உள்ள பத்தியைப் படித்து கேள்விக்குப் பதில் அளிக்கவும்.\n\nபத்தி: {context}\n\nகேள்வி: {question}"
# "The answer to this question is not in this passage."
DEFAULT_UNANSWERABLE = "இந்தப் பத்தியில் இந்தக் கேள்விக்கான பதில் இல்லை."

_ROLE_MAP = {"user": "user", "human": "user", "prompter": "user", "assistant": "assistant", "gpt": "assistant",
             "model": "assistant", "bot": "assistant", "system": "system"}


def _field(row: Mapping[str, Any], fields: Mapping[str, str], name: str, default: Any = "") -> Any:
    return row.get(fields.get(name, name), default)


def row_to_messages(row: Mapping[str, Any], fmt: str, fields: Mapping[str, str],
                    options: Mapping[str, Any] | None = None) -> list[dict[str, str]] | None:
    options = options or {}
    if fmt == "alpaca":
        instr = (_field(row, fields, "instruction") or "").strip()
        inp = (_field(row, fields, "input") or "").strip()
        out = (_field(row, fields, "output") or "").strip()
        if not instr or not out:
            return None
        return [{"role": "user", "content": f"{instr}\n\n{inp}" if inp else instr},
                {"role": "assistant", "content": out}]
    if fmt == "prompt_response":
        p = (_field(row, fields, "prompt") or "").strip()
        r = (_field(row, fields, "response") or "").strip()
        return [{"role": "user", "content": p}, {"role": "assistant", "content": r}] if p and r else None
    if fmt == "qa":
        ctx = strip_prefixes(_field(row, fields, "context") or "", options.get("strip_prefixes"))
        q = (_field(row, fields, "question") or "").strip()
        a = (_field(row, fields, "answer") or "").strip()
        if not ctx or not q:
            return None
        if not a:
            a = options.get("unanswerable_response", DEFAULT_UNANSWERABLE)
            if not a:
                return None
        prompt = options.get("qa_template", DEFAULT_QA_TEMPLATE).format(context=ctx, question=q)
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": a}]
    if fmt in ("messages", "sharegpt"):
        key = "messages" if fmt == "messages" else "conversations"
        turns = _field(row, fields, key, None) or []
        role_k, content_k = ("role", "content") if fmt == "messages" else ("from", "value")
        msgs = []
        for t in turns:
            role = _ROLE_MAP.get(str(t.get(role_k, "")).lower())
            content = (t.get(content_k) or "").strip()
            if role and content:
                msgs.append({"role": role, "content": content})
        return msgs if any(m["role"] == "assistant" for m in msgs) else None
    raise ValueError(f"Unknown SFT format {fmt!r}")


def iter_conversations(sources: list[Mapping[str, Any]], normalizer: TextNormalizer | None,
                       system_prompt: str | None = None) -> Iterator[dict[str, Any]]:
    for src in sources:
        if not src.get("enabled", True):
            continue
        fmt, fields = src.get("format", "alpaca"), dict(src.get("fields") or {})
        n = 0
        try:
            for row in iter_rows(src):
                msgs = row_to_messages(row, fmt, fields, src)
                if not msgs:
                    continue
                if normalizer is not None:
                    msgs = [{**m, "content": normalizer(m["content"])} for m in msgs]
                sp = src.get("system_prompt", system_prompt)
                if sp and msgs[0]["role"] != "system":
                    msgs = [{"role": "system", "content": sp}] + msgs
                n += 1
                yield {"messages": msgs, "source": src["name"]}
        except Exception as exc:
            if src.get("required", False):
                raise
            log.error("SFT source %s failed and was skipped: %s: %s", src["name"], type(exc).__name__, exc)
        log.info("SFT source %s: %d conversations", src["name"], n)


def encode_conversation(tok, messages: list[dict[str, str]], max_len: int,
                        truncation: str = "drop") -> dict[str, list[int]] | None:
    """Token ids + labels (assistant spans only). Returns None if nothing trainable remains."""
    msgs = list(messages)
    prefix = ""
    if msgs and msgs[0]["role"] == "system":
        prefix = msgs[0]["content"].strip() + "\n\n"
        msgs = msgs[1:]
    ids: list[int] = [tok.bos_token_id]
    labels: list[int] = [IGNORE]

    def enc(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    first_user = True
    for m in msgs:
        content = m["content"].strip()
        if m["role"] == "user":
            if first_user:
                content, first_user = prefix + content, False
            seg = enc(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            ids += seg
            labels += [IGNORE] * len(seg)
        elif m["role"] == "assistant":
            head = enc("<start_of_turn>model\n")
            body = enc(f"{content}<end_of_turn>")
            tail = enc("\n")
            ids += head + body + tail
            labels += [IGNORE] * len(head) + body + [IGNORE] * len(tail)
    if len(ids) > max_len:
        if truncation == "drop":
            return None
        ids, labels = ids[:max_len], labels[:max_len]
    if all(x == IGNORE for x in labels):
        return None
    return {"input_ids": ids, "labels": labels}


@dataclass
class ChatCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        n = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            n = ((n + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        ids = torch.full((len(features), n), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), n), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(features), n), dtype=torch.long)
        for i, f in enumerate(features):
            L = len(f["input_ids"])
            ids[i, :L] = torch.tensor(f["input_ids"])
            labels[i, :L] = torch.tensor(f["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
