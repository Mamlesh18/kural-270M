"""Hierarchical YAML configuration.

Every entrypoint takes ``--config path/to/file.yaml`` plus optional ``key=value``
overrides (OmegaConf dot-list syntax). A config file may declare::

    defaults:
      - ../base.yaml          # paths are relative to the file that declares them
      - ../model/gemma3_270m.yaml

Parents are merged first (left to right), then the file itself, then CLI
overrides. Interpolation (``${paths.workspace}``) and environment lookups
(``${oc.env:VAR,default}``) are resolved after merging, so child files can
change values that parents interpolate.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("repo_root"):
        OmegaConf.register_new_resolver("repo_root", lambda: REPO_ROOT.as_posix())
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda *xs: _prod(xs))
    if not OmegaConf.has_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda a, b: a / b)


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


_register_resolvers()


def _load_tree(path: Path, _stack: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in _stack:
        chain = " -> ".join(p.name for p in (*_stack, path))
        raise ValueError(f"Cyclic config defaults: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    node = OmegaConf.load(path)
    if not isinstance(node, DictConfig):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    parents = node.pop("defaults", None) or []
    if isinstance(parents, (str,)):
        parents = [parents]
    merged = OmegaConf.create({})
    for parent in parents:
        merged = OmegaConf.merge(merged, _load_tree(path.parent / str(parent), (*_stack, path)))
    return OmegaConf.merge(merged, node)


def load_config(path: str | os.PathLike, overrides: Sequence[str] | None = None,
                resolve: bool = True) -> DictConfig:
    cfg = _load_tree(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    if resolve:
        OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def to_container(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides, e.g. train.lr=1e-4")
    return parser


def parse_config(description: str, argv: Sequence[str] | None = None) -> DictConfig:
    args = build_arg_parser(description).parse_args(argv)
    return load_config(args.config, args.overrides)


def git_info() -> dict[str, Any]:
    def _run(*cmd: str) -> str | None:
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    status = _run("git", "status", "--porcelain")
    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "datasets", "tokenizers", "sentencepiece", "accelerate"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def save_run_metadata(cfg: DictConfig, out_dir: str | os.PathLike, name: str = "run") -> Path:
    """Persist the fully-resolved config + environment next to the outputs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / f"{name}_config.yaml", resolve=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "git": git_info(),
        "versions": _versions(),
        "platform": platform.platform(),
    }
    (out / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
