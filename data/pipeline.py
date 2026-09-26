"""Corpus ingestion & cleaning pipeline.

    python -m data.pipeline --config configs/data/pipeline.yaml [key=value ...]

For every enabled source, documents flow through::

    read → normalize → line filter → language ID → route to category
         → quality filter → exact/near dedup (global, first wins) → train/eval split → shard
         → (optional) synthetic Tanglish / spoken-Tamil augmentation of Tamil train docs

Output layout (``cfg.output_dir``)::

    <category>/train-00000.jsonl[.gz]   records: {id, text, source, source_id, category, lang, ...}
    <category>/eval-00000.jsonl[.gz]
    stats.json                          per-source rejection reasons, per-category sizes
    pipeline_config.yaml / pipeline_metadata.json

CPU-heavy steps (normalization, LID, quality, MinHash) run in ``num_workers``
processes; the dedup index and writers stay in the main process so results are
deterministic for a given source order.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from common.config import parse_config, save_run_metadata, to_container
from common.utils import ShardedJsonlWriter, get_logger, set_seed, setup_logging, stable_fraction, text_hash
from data.dedup import Deduplicator
from data.langid import LanguageIdentifier
from data.normalize import TextNormalizer
from data.quality import LineFilter, QualityFilter
from data.sources import iter_source
from data.tanglish import Augmenter

log = get_logger("data.pipeline")

DEFAULT_LANG_CATEGORY = {"ta": "ta_web", "code_mixed": "code_mixed", "tanglish": "tanglish", "en": "en"}


class DocProcessor:
    """Stateless per-document work (safe to run in worker processes)."""

    def __init__(self, cfg: dict[str, Any]):
        self.normalize = TextNormalizer(cfg.get("normalize"))
        self.line_filter = LineFilter(cfg.get("line_filter"))
        self.langid = LanguageIdentifier(cfg.get("langid"))
        self.quality = QualityFilter(cfg.get("quality"))
        dedup_cfg = cfg.get("dedup") or {}
        self.dedup = Deduplicator(dedup_cfg) if dedup_cfg.get("near", True) else None
        self.sources = {s["name"]: s for s in cfg["sources"]}

    def __call__(self, doc: dict[str, Any]) -> dict[str, Any]:
        src = self.sources[doc["source"]]
        text = self.normalize(doc["text"])
        text = self.line_filter(text)
        if not text:
            return {"status": "reject", "reason": "empty_after_cleaning", "source": doc["source"]}
        lang = self.langid(text)
        allowed = src.get("allowed_langs") or [src.get("lang_hint", "ta")]
        if lang.label not in allowed:
            return {"status": "reject", "reason": f"lang_{lang.label}", "source": doc["source"],
                    "lang": lang.label}
        profile = src.get("quality_profile") or lang.label
        ok, reason, _ = self.quality.check(text, profile)
        if not ok:
            return {"status": "reject", "reason": f"quality_{reason}", "source": doc["source"],
                    "lang": lang.label}
        if lang.label == src.get("lang_hint"):
            category = src["category"]
        else:
            category = (src.get("category_by_lang") or {}).get(lang.label, DEFAULT_LANG_CATEGORY.get(lang.label, lang.label))
        sig = self.dedup.signature(text) if self.dedup is not None else None
        record = {
            "id": text_hash(text),
            "text": text,
            "source": doc["source"],
            "source_id": doc["source_id"],
            "category": category,
            **lang.as_dict(),
        }
        return {"status": "ok", "record": record, "signature": sig, "source": doc["source"]}


_WORKER: DocProcessor | None = None


def _worker_init(cfg: dict[str, Any]) -> None:
    global _WORKER
    setup_logging("WARNING")
    _WORKER = DocProcessor(cfg)


def _worker_run(doc: dict[str, Any]) -> dict[str, Any]:
    assert _WORKER is not None
    return _WORKER(doc)


def _iter_all_sources(sources: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for src in sources:
        log.info("Reading source %s (%s: %s)", src["name"], src.get("type", "hf"), src["path"])
        try:
            yield from iter_source(src)
        except Exception as exc:  # keep going; a bad source should not kill a long run
            if src.get("required", False):
                raise
            log.error("Source %s failed and was skipped: %s: %s", src["name"], type(exc).__name__, exc)


def _prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty; pass overwrite=true to replace it")
        if not (out / "stats.json").exists():
            raise FileExistsError(f"Refusing to delete {out}: it does not look like a pipeline output")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def run(cfg) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 0)))
    out = Path(cfg.output_dir)
    _prepare_output(out, bool(cfg.get("overwrite", False)))
    save_run_metadata(cfg, out, "pipeline")
    c = to_container(cfg)
    c["sources"] = [s for s in c["sources"] if s.get("enabled", True)]
    if not c["sources"]:
        raise ValueError("No enabled sources in config")
    names = [s["name"] for s in c["sources"]]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate source names: {names}")

    dedup = Deduplicator(c.get("dedup"))
    split = c.get("split") or {}
    eval_fraction = float(split.get("eval_fraction", 0.01))
    eval_max = int(split.get("eval_max_docs_per_category", 2000))
    seed_salt = str(c.get("seed", 0))
    augment = Augmenter(c.get("augment"))
    aug_langs = set((c.get("augment") or {}).get("source_langs", ["ta"]))
    shard_cfg = c.get("sharding") or {}
    docs_per_shard = int(shard_cfg.get("docs_per_shard", 50_000))
    compress = bool(shard_cfg.get("compress", False))

    writers: dict[tuple[str, str], ShardedJsonlWriter] = {}

    def writer(category: str, split_name: str) -> ShardedJsonlWriter:
        key = (category, split_name)
        if key not in writers:
            writers[key] = ShardedJsonlWriter(out / category, split_name, docs_per_shard, compress)
        return writers[key]

    src_stats: dict[str, Counter] = defaultdict(Counter)
    cat_stats: dict[str, Counter] = defaultdict(Counter)
    eval_counts: Counter = Counter()

    num_workers = int(c.get("num_workers", 1))
    docs = _iter_all_sources(c["sources"])
    pool = None
    if num_workers > 1:
        pool = mp.get_context("spawn").Pool(num_workers, initializer=_worker_init, initargs=(c,))
        results = pool.imap(_worker_run, docs, chunksize=int(c.get("chunksize", 64)))
    else:
        proc = DocProcessor(c)
        results = map(proc, docs)

    t0 = time.time()
    log_every = int(c.get("log_every", 5000))
    n = 0
    try:
        for res in results:
            n += 1
            s = src_stats[res["source"]]
            s["seen"] += 1
            if res["status"] != "ok":
                s[res["reason"]] += 1
            else:
                rec = res["record"]
                dup = dedup.is_duplicate(rec["text"], res["signature"])
                if dup:
                    s[f"dup_{dup}"] += 1
                else:
                    s["kept"] += 1
                    s[f"lang_kept_{rec['lang']}"] += 1
                    cat = rec["category"]
                    is_eval = (stable_fraction(rec["id"], seed_salt) < eval_fraction
                               and eval_counts[cat] < eval_max)
                    split_name = "eval" if is_eval else "train"
                    if is_eval:
                        eval_counts[cat] += 1
                    writer(cat, split_name).write(rec)
                    cat_stats[cat][f"{split_name}_docs"] += 1
                    cat_stats[cat][f"{split_name}_chars"] += len(rec["text"])
                    if not is_eval and rec["lang"] in aug_langs:
                        for aug_cat, aug_text in augment(rec["text"]):
                            aug = {**rec, "id": text_hash(aug_text), "text": aug_text,
                                   "category": aug_cat, "synthetic": True, "parent_id": rec["id"]}
                            writer(aug_cat, "train").write(aug)
                            cat_stats[aug_cat]["train_docs"] += 1
                            cat_stats[aug_cat]["train_chars"] += len(aug_text)
            if n % log_every == 0:
                kept = sum(v["kept"] for v in src_stats.values())
                log.info("%d docs processed | %d kept | %.0f docs/s", n, kept, n / (time.time() - t0))
    finally:
        if pool is not None:
            pool.terminate()
        for w in writers.values():
            w.close()

    stats = {
        "docs_processed": n,
        "elapsed_sec": round(time.time() - t0, 1),
        "dedup": dedup.stats,
        "sources": {k: dict(v) for k, v in src_stats.items()},
        "categories": {k: dict(v) for k, v in sorted(cat_stats.items())},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Done: %d docs in %.1fs", n, stats["elapsed_sec"])
    for cat, v in stats["categories"].items():
        log.info("  %-22s train=%7d docs %10d chars | eval=%5d docs", cat, v.get("train_docs", 0),
                 v.get("train_chars", 0), v.get("eval_docs", 0))
    return stats


def main(argv=None) -> None:
    setup_logging()
    cfg = parse_config("Tamil corpus ingestion and cleaning pipeline", argv)
    run(cfg)


if __name__ == "__main__":
    main()
