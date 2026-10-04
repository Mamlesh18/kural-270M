"""Extend the Gemma 3 tokenizer with Tamil pieces.

    python -m tokenizer.adapt --config configs/tokenizer/extend_gemma.yaml

Strategy (``extend``): keep every Gemma piece and id, so English/code/other
languages encode exactly as before, and add Tamil pieces learned by our Tamil BPE:

* Only pieces made purely of Tamil script (optionally with the ``▁`` space marker)
  are taken, so non-Tamil text is unaffected.
* Gemma 3 ships ~6k ``<unusedN>`` placeholder slots; new pieces fill those first
  (``reuse_unused``), so up to ~6k Tamil pieces cost *zero* extra parameters.
  Remaining pieces are appended (embedding matrix grows).
* ``score_policy: tamil_first`` re-scores all Tamil-only pieces (new *and*
  overlapping) in the Tamil BPE's merge order, above Gemma's own merges. Tamil
  text then segments (almost) exactly like the Tamil tokenizer; non-Tamil text is
  untouched. ``append`` puts new pieces after all Gemma merges instead.
* Taking the top-N Tamil pieces by BPE rank preserves merge closure (every
  piece's parents rank higher), so every added piece is reachable.

Outputs a SentencePiece ``tokenizer.model`` (for GGUF/llama.cpp), a
``GemmaTokenizerFast`` (for HF), and ``extension_manifest.json`` listing the new
ids, which the model factory uses to initialize their embeddings.

The ``replace`` strategy (use the Tamil/bilingual tokenizer as the whole vocab)
needs no tokenizer surgery; see ``training/model_factory.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from common.config import parse_config, save_run_metadata, to_container
from common.tamil import is_tamil_piece
from common.utils import ensure_dir, get_logger, read_jsonl, setup_logging
from tokenizer.train_tokenizer import DEFAULT_CHAT_TEMPLATE, available_categories, category_files, export_hf

log = get_logger("tokenizer.adapt")

_UNUSED_RE = re.compile(r"^<unused\d+>$")


def _load_proto(path: str | Path):
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(Path(path).read_bytes())
    return m


def resolve_base_sp_model(name_or_path: str, revision: str | None = None) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p
    if p.is_dir() and (p / "tokenizer.model").exists():
        return p / "tokenizer.model"
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(name_or_path, "tokenizer.model", revision=revision))


def extend_sp_model(base_path: Path, tamil_path: Path, *, reuse_unused: bool = True,
                    max_new_tokens: int | None = None, min_piece_chars: int = 1,
                    score_policy: str = "tamil_first"):
    """Returns (merged ModelProto, manifest dict)."""
    from sentencepiece import sentencepiece_model_pb2 as pb

    NORMAL = pb.ModelProto.SentencePiece.NORMAL
    base = _load_proto(base_path)
    tamil = _load_proto(tamil_path)
    if base.trainer_spec.model_type != pb.TrainerSpec.BPE:
        raise ValueError("extend requires a BPE base tokenizer (Gemma 3 is BPE)")
    if tamil.trainer_spec.model_type != pb.TrainerSpec.BPE:
        raise ValueError("extend requires the Tamil tokenizer to be BPE (model_type: bpe) for merge closure")

    base_index = {p.piece: i for i, p in enumerate(base.pieces)}
    tamil_normal = [p for p in tamil.pieces if p.type == NORMAL and is_tamil_piece(p.piece)
                    and len(p.piece.replace("▁", "")) >= min_piece_chars]
    # BPE: higher score = earlier merge. Keep Tamil merge order.
    tamil_normal.sort(key=lambda p: -p.score)
    new = [p for p in tamil_normal if p.piece not in base_index]
    if max_new_tokens is not None and len(new) > max_new_tokens:
        cutoff_piece = new[max_new_tokens - 1].piece
        keep = {p.piece for p in new[:max_new_tokens]}
        # Truncate the overlap list at the same rank so re-scored pieces stay consistent.
        rank = {p.piece: r for r, p in enumerate(tamil_normal)}
        tamil_normal = [p for p in tamil_normal if rank[p.piece] <= rank[cutoff_piece]]
        new = [p for p in new if p.piece in keep]

    normal_scores = [p.score for p in base.pieces if p.type == NORMAL]
    base_max, base_min = max(normal_scores), min(normal_scores)
    n_tamil = len(tamil_normal)
    if score_policy == "tamil_first":
        # Rank r (0 = first merge) → score strictly above every Gemma merge.
        tamil_score = {p.piece: base_max + float(n_tamil - r) for r, p in enumerate(tamil_normal)}
    elif score_policy == "append":
        tamil_score = {p.piece: base_min - 1.0 - float(r) for r, p in enumerate(new)}
    else:
        raise ValueError(f"Unknown score_policy {score_policy!r}")

    merged = _load_proto(base_path)
    rescored = 0
    if score_policy == "tamil_first":
        for p in merged.pieces:
            if p.type == NORMAL and p.piece in tamil_score:
                p.score = tamil_score[p.piece]
                rescored += 1

    unused_slots = [i for i, p in enumerate(merged.pieces) if _UNUSED_RE.match(p.piece)] if reuse_unused else []
    reused_ids: list[int] = []
    appended_ids: list[int] = []
    for k, p in enumerate(new):
        score = tamil_score[p.piece]
        if k < len(unused_slots):
            slot = merged.pieces[unused_slots[k]]
            slot.piece, slot.score, slot.type = p.piece, score, NORMAL
            reused_ids.append(unused_slots[k])
        else:
            sp = merged.pieces.add()
            sp.piece, sp.score, sp.type = p.piece, score, NORMAL
            appended_ids.append(len(merged.pieces) - 1)

    manifest = {
        "strategy": "extend",
        "score_policy": score_policy,
        "base_vocab_size": len(base.pieces),
        "merged_vocab_size": len(merged.pieces),
        "tamil_tamil_only_pieces": n_tamil,
        "rescored_overlapping_pieces": rescored,
        "num_new_pieces": len(new),
        "reused_unused_slots": len(reused_ids),
        "appended": len(appended_ids),
        "new_token_ids": reused_ids + appended_ids,
        "new_pieces": [p.piece for p in new],
    }
    return merged, manifest


def _sample_texts(processed_dir: Path, categories: list[str], per_category: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    present = set(available_categories(processed_dir)) if processed_dir.exists() else set()
    for cat in categories:
        if cat not in present:
            continue
        files = category_files(processed_dir, cat, "eval") or category_files(processed_dir, cat, "train")
        texts: list[str] = []
        for f in files:
            for row in read_jsonl(f):
                texts.append(row["text"][:3000])
                if len(texts) >= per_category:
                    break
            if len(texts) >= per_category:
                break
        out[cat] = texts
    return out


def compare_encodings(base_tok, new_tok, samples: dict[str, list[str]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for cat, texts in samples.items():
        if not texts:
            continue
        b = [base_tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        n = [new_tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        report[cat] = {
            "base_tokens": sum(map(len, b)),
            "extended_tokens": sum(map(len, n)),
            "token_reduction": round(1 - sum(map(len, n)) / max(sum(map(len, b)), 1), 4),
            "identical_encoding_rate": round(sum(x == y for x, y in zip(b, n)) / len(texts), 4),
            "roundtrip_exact": round(sum(new_tok.decode(y) == t for y, t in zip(n, texts)) / len(texts), 4),
        }
    return report


def run(cfg) -> Path:
    from transformers import AutoTokenizer

    acfg = to_container(cfg.adapt)
    out_dir = ensure_dir(acfg["output_dir"])
    save_run_metadata(cfg, out_dir, "kural_adapt")
    base_name = acfg.get("base_tokenizer") or cfg.base_model.name_or_path
    revision = cfg.base_model.get("revision")
    base_sp = resolve_base_sp_model(base_name, revision)
    tamil_sp = Path(acfg["tamil_tokenizer_dir"]) / "tokenizer.model"
    log.info("Extending %s with Tamil pieces from %s", base_sp, tamil_sp)
    merged, manifest = extend_sp_model(
        base_sp, tamil_sp, reuse_unused=bool(acfg.get("reuse_unused", True)),
        max_new_tokens=acfg.get("max_new_tokens"), min_piece_chars=int(acfg.get("min_piece_chars", 1)),
        score_policy=acfg.get("score_policy", "tamil_first"))
    work = ensure_dir(out_dir / "_work")
    merged_path = work / "merged.model"
    merged_path.write_bytes(merged.SerializeToString())
    log.info("New pieces: %d (reused %d unused slots, appended %d); vocab %d → %d",
             manifest["num_new_pieces"], manifest["reused_unused_slots"], manifest["appended"],
             manifest["base_vocab_size"], manifest["merged_vocab_size"])

    base_tok = AutoTokenizer.from_pretrained(base_name, revision=revision)
    log.info("Converting merged SentencePiece model to a fast tokenizer (takes a minute for 262k pieces)...")
    new_tok = export_hf(merged_path, out_dir, base_tok.chat_template or DEFAULT_CHAT_TEMPLATE,
                        int(acfg.get("model_max_length", 32768)))
    merged_path.unlink()
    work.rmdir()

    samples = _sample_texts(Path(acfg["processed_dir"]), list(acfg.get("report_categories", [])),
                            int(acfg.get("report_docs_per_category", 200)))
    manifest["encoding_report"] = compare_encodings(base_tok, new_tok, samples)
    for cat, r in manifest["encoding_report"].items():
        log.info("  %-20s tokens %8d → %8d (%.1f%% fewer) | identical=%.3f roundtrip=%.3f", cat,
                 r["base_tokens"], r["extended_tokens"], 100 * r["token_reduction"],
                 r["identical_encoding_rate"], r["roundtrip_exact"])
    (out_dir / "extension_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")
    return out_dir


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Extend the Gemma tokenizer with Tamil pieces", argv))


if __name__ == "__main__":
    main()
