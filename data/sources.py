"""Corpus source readers.

A source is a mapping in the pipeline config::

    - name: wiki_ta                 # unique id, stored on every record
      type: hf                      # hf | jsonl | text | parquet
      path: wikimedia/wikipedia     # HF repo id, or a local file / directory / glob
      config_name: 20231101.ta      # hf only (optional)
      data_dir: null                # hf only (optional)
      data_files: null              # hf only (optional)
      split: train
      streaming: true
      text_field: text              # or a list of fields joined with "\\n\\n"
      id_field: null
      max_docs: null
      skip_docs: 0
      filter: {field: language, equals: Tamil}   # optional row filter
      strip_prefixes: ["சூழல்:"]  # optional labels removed from the start of the text
      shuffle_seed: null            # hf only: shuffle before max_docs to take a random sample
      category: ta_wiki             # mixture bucket for docs detected as lang_hint
      lang_hint: ta                 # expected language label
      allowed_langs: [ta, code_mixed]
      enabled: true

Readers yield ``{"text", "source", "source_id"}`` dicts.
"""

from __future__ import annotations

import glob
import itertools
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from common.utils import get_logger, read_jsonl

log = get_logger(__name__)


def _expand(path: str) -> list[Path]:
    p = Path(os.path.expanduser(os.path.expandvars(path)))
    if p.is_dir():
        files = sorted(x for x in p.rglob("*") if x.is_file())
    elif any(ch in path for ch in "*?["):
        files = sorted(Path(x) for x in glob.glob(str(p), recursive=True))
    else:
        files = [p]
    missing = [f for f in files if not f.exists()]
    if missing or not files:
        raise FileNotFoundError(f"No input files for {path!r}")
    return files


def _get_text(row: Mapping[str, Any], field: str | list[str]) -> str:
    if isinstance(field, str):
        val = row.get(field)
        return val if isinstance(val, str) else ("" if val is None else str(val))
    parts = [row.get(f) for f in field]
    return "\n\n".join(p for p in parts if isinstance(p, str) and p.strip())


def strip_prefixes(text: str, prefixes: list[str] | None) -> str:
    """Remove a leading label such as "சூழல்:" ("Context:") left over from dataset templates."""
    text = text.lstrip()
    for p in prefixes or []:
        if text.startswith(p):
            return text[len(p):].lstrip()
    return text


def _row_filter(spec: Mapping[str, Any] | None):
    if not spec:
        return lambda row: True
    field = spec["field"]
    if "equals" in spec:
        return lambda row: row.get(field) == spec["equals"]
    if "in" in spec:
        allowed = set(spec["in"])
        return lambda row: row.get(field) in allowed
    raise ValueError(f"Unsupported filter spec: {dict(spec)}")


def _iter_hf(src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": src.get("split", "train"), "streaming": src.get("streaming", True)}
    for key in ("data_dir", "data_files", "revision"):
        if src.get(key) is not None:
            kwargs[key] = src[key]
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    ds = load_dataset(src["path"], src.get("config_name"), **kwargs)
    if src.get("shuffle_seed") is not None:
        # Random sample instead of the first rows (with max_docs). Streaming datasets are
        # shuffled through a buffer, so the sample is approximate.
        seed = int(src["shuffle_seed"])
        ds = ds.shuffle(seed=seed, buffer_size=10_000) if kwargs["streaming"] else ds.shuffle(seed=seed)
    yield from ds


def _iter_text(src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    mode = src.get("doc_sep", "blank_line")  # line | blank_line | file
    for f in _expand(src["path"]):
        content = f.read_text(encoding="utf-8", errors="replace")
        if mode == "file":
            docs = [content]
        elif mode == "line":
            docs = content.splitlines()
        else:
            docs = [d for d in content.replace("\r\n", "\n").split("\n\n")]
        for i, d in enumerate(docs):
            if d.strip():
                yield {"text": d, "id": f"{f.name}:{i}"}


def _iter_jsonl(src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for f in _expand(src["path"]):
        yield from read_jsonl(f)


def _iter_parquet(src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for f in _expand(src["path"]):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1024):
            yield from batch.to_pylist()


_READERS = {"hf": _iter_hf, "jsonl": _iter_jsonl, "text": _iter_text, "parquet": _iter_parquet}


def iter_rows(src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Raw rows of a source after ``filter`` / ``skip_docs`` / ``max_docs``."""
    kind = src.get("type", "hf")
    if kind not in _READERS:
        raise ValueError(f"Unknown source type {kind!r} for {src.get('name')}")
    rows = _READERS[kind](src)
    keep = _row_filter(src.get("filter"))
    rows = (r for r in rows if keep(r))
    skip = int(src.get("skip_docs") or 0)
    max_docs = src.get("max_docs")
    return itertools.islice(rows, skip, None if max_docs is None else skip + int(max_docs))


def iter_source(src: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    skip = int(src.get("skip_docs") or 0)
    text_field = src.get("text_field", "text")
    id_field = src.get("id_field")
    prefixes = src.get("strip_prefixes")
    for i, row in enumerate(iter_rows(src)):
        text = strip_prefixes(_get_text(row, text_field), prefixes)
        if not text:
            continue
        yield {
            "text": text,
            "source": src["name"],
            "source_id": str(row.get(id_field)) if id_field else str(row.get("id", i + skip)),
        }
