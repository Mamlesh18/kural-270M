"""Small shared helpers: logging, seeding, JSONL shards, hashing, device checks."""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import xxhash

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    # Windows consoles default to cp1252, which cannot print Tamil.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S", force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def text_hash(text: str) -> str:
    return xxhash.xxh64_hexdigest(text.encode("utf-8"))


def stable_fraction(key: str, salt: str = "") -> float:
    """Deterministic value in [0, 1) derived from a key (for reproducible splits)."""
    return xxhash.xxh64_intdigest((salt + key).encode("utf-8")) / 2**64


def _open(path: Path, mode: str):
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, mode.replace("t", "") + "b"), encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    with _open(Path(path), "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


class ShardedJsonlWriter:
    """Writes ``{prefix}-00000.jsonl[.gz]`` shards of at most ``docs_per_shard`` rows."""

    def __init__(self, directory: str | os.PathLike, prefix: str = "train",
                 docs_per_shard: int = 50_000, compress: bool = False):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.docs_per_shard = docs_per_shard
        self.ext = ".jsonl.gz" if compress else ".jsonl"
        self._shard = 0
        self._count = 0
        self._fh = None
        self.total = 0

    def _roll(self) -> None:
        if self._fh:
            self._fh.close()
        path = self.dir / f"{self.prefix}-{self._shard:05d}{self.ext}"
        self._fh = _open(path, "wt")
        self._shard += 1
        self._count = 0

    def write(self, row: dict[str, Any]) -> None:
        if self._fh is None or self._count >= self.docs_per_shard:
            self._roll()
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._count += 1
        self.total += 1

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def resolve_precision(setting: str | bool) -> dict[str, bool]:
    """Map ``precision: auto|bf16|fp16|fp32`` to TrainingArguments flags."""
    import torch

    setting = str(setting).lower()
    cuda = torch.cuda.is_available()
    if setting == "auto":
        if cuda and torch.cuda.is_bf16_supported():
            setting = "bf16"
        elif cuda:
            setting = "fp16"
        else:
            setting = "fp32"
    if setting == "bf16" and cuda and not torch.cuda.is_bf16_supported():
        raise RuntimeError("precision=bf16 requested but this GPU does not support bf16")
    return {"bf16": setting == "bf16", "fp16": setting == "fp16"}


def torch_dtype(name: str | None):
    import torch

    if name in (None, "auto"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[name]


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
