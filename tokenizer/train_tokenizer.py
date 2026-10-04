"""Train a Tamil-focused SentencePiece tokenizer and export it as a Hugging Face fast tokenizer.

    python -m tokenizer.train_tokenizer --config configs/tokenizer/kural_bpe_32k.yaml

Steps:
  1. Sample training text from processed shards with per-category character budgets
     (so English / Tanglish / spoken Tamil are represented in known proportions).
  2. Train SentencePiece with Gemma-compatible settings (identity normalization — we
     already NFC-normalize —, no dummy prefix, byte fallback, split digits, and
     Gemma's special-token ids: <pad>=0 <eos>=1 <bos>=2 <unk>=3).
  3. Convert to ``GemmaTokenizerFast`` so the result is a drop-in replacement for the
     Gemma tokenizer (same special tokens, same chat markers).
  4. Validate: HF vs SentencePiece agreement and decode round-trip on held-out text.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterator

from common.config import parse_config, save_run_metadata, to_container
from common.utils import ensure_dir, get_logger, read_jsonl, set_seed, setup_logging

log = get_logger("tokenizer.train")

GEMMA_SPECIALS = {"pad": (0, "<pad>"), "eos": (1, "<eos>"), "bos": (2, "<bos>"), "unk": (3, "<unk>")}
DEFAULT_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "{%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}{%- set first_user_prefix = '' -%}{%- set loop_messages = messages -%}{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}"
    "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else '') + message['content'] | trim + '<end_of_turn>\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\n' }}{%- endif -%}"
)


def category_files(processed_dir: Path, category: str, split: str = "train") -> list[Path]:
    d = processed_dir / category
    return sorted(d.glob(f"{split}-*.jsonl*")) if d.is_dir() else []


def available_categories(processed_dir: Path) -> list[str]:
    return sorted(p.name for p in processed_dir.iterdir() if p.is_dir() and category_files(processed_dir, p.name))


def iter_category_docs(processed_dir: Path, category: str, seed: int, split: str = "train") -> Iterator[str]:
    files = category_files(processed_dir, category, split)
    random.Random(seed).shuffle(files)
    for f in files:
        for row in read_jsonl(f):
            yield row["text"]


def write_sample(processed_dir: Path, weights: dict[str, float], total_chars: int, out_file: Path,
                 seed: int, max_line_chars: int) -> dict[str, Any]:
    present = set(available_categories(processed_dir))
    missing = [c for c in weights if c not in present]
    if missing:
        log.warning("Tokenizer sample: categories without data are skipped: %s", missing)
    weights = {c: w for c, w in weights.items() if c in present and w > 0}
    if not weights:
        raise ValueError(f"No sample categories available in {processed_dir} (have: {sorted(present)})")
    z = sum(weights.values())
    budgets = {c: int(total_chars * w / z) for c, w in weights.items()}
    report: dict[str, Any] = {}
    with open(out_file, "w", encoding="utf-8") as f:
        for cat, budget in budgets.items():
            written = lines = 0
            for doc in iter_category_docs(processed_dir, cat, seed):
                for line in doc.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    line = line[:max_line_chars]
                    f.write(line + "\n")
                    written += len(line)
                    lines += 1
                if written >= budget:
                    break
            report[cat] = {"budget_chars": budget, "chars": written, "lines": lines}
            if written < budget:
                log.warning("Category %s: only %d/%d chars available", cat, written, budget)
    total = sum(r["chars"] for r in report.values())
    for cat, r in report.items():
        r["share"] = round(r["chars"] / max(total, 1), 4)
        log.info("  sample %-22s %12d chars (%5.1f%%)", cat, r["chars"], 100 * r["share"])
    return report


def train_sentencepiece(sample_file: Path, model_prefix: Path, tcfg: dict[str, Any]) -> Path:
    import sentencepiece as spm

    user_symbols = list(tcfg.get("user_defined_symbols") or [])
    args: dict[str, Any] = dict(
        input=str(sample_file),
        model_prefix=str(model_prefix),
        model_type=tcfg.get("model_type", "bpe"),
        vocab_size=int(tcfg["vocab_size"]),
        character_coverage=float(tcfg.get("character_coverage", 0.99995)),
        byte_fallback=bool(tcfg.get("byte_fallback", True)),
        split_digits=bool(tcfg.get("split_digits", True)),
        split_by_unicode_script=True,
        split_by_whitespace=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        normalization_rule_name="identity",
        max_sentence_length=int(tcfg.get("max_sentence_length", 16384)),
        input_sentence_size=int(tcfg.get("input_sentence_size", 10_000_000)),
        shuffle_input_sentence=True,
        num_threads=int(tcfg.get("num_threads", 8)),
        seed_sentencepiece_size=int(tcfg.get("seed_sentencepiece_size", 1_000_000)),
        train_extremely_large_corpus=bool(tcfg.get("train_extremely_large_corpus", False)),
        hard_vocab_limit=bool(tcfg.get("hard_vocab_limit", True)),
        pad_id=0, eos_id=1, bos_id=2, unk_id=3,
        pad_piece="<pad>", eos_piece="<eos>", bos_piece="<bos>", unk_piece="<unk>",
        user_defined_symbols=user_symbols,
        minloglevel=int(tcfg.get("minloglevel", 1)),
    )
    args.update(tcfg.get("extra_trainer_args") or {})
    log.info("Training SentencePiece (%s, vocab=%d)...", args["model_type"], args["vocab_size"])
    spm.SentencePieceTrainer.train(**args)
    return model_prefix.with_suffix(".model")


def export_hf(sp_model: Path, out_dir: Path, chat_template: str | None = DEFAULT_CHAT_TEMPLATE,
              model_max_length: int = 32768):
    """Wrap a Gemma-convention SentencePiece model as a GemmaTokenizerFast and save it."""
    from transformers import GemmaTokenizerFast

    tok = GemmaTokenizerFast(
        vocab_file=str(sp_model), bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
        pad_token="<pad>", add_bos_token=True, add_eos_token=False, from_slow=True,
    )
    tok.model_max_length = model_max_length
    if chat_template:
        tok.chat_template = chat_template
    extra = [t for t in ("<start_of_turn>", "<end_of_turn>") if t in tok.get_vocab()]
    if extra:
        tok.add_special_tokens({"additional_special_tokens": extra})
    tok.save_pretrained(str(out_dir))
    shutil.copy(sp_model, out_dir / "tokenizer.model")
    return tok


def validate(tok, sp_model: Path, texts: list[str]) -> dict[str, float]:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(sp_model))
    agree = roundtrip = 0
    n_tok = n_bytefb = 0
    for t in texts:
        hf_ids = tok(t, add_special_tokens=False)["input_ids"]
        agree += hf_ids == sp.encode(t)
        roundtrip += tok.decode(hf_ids) == t
        n_tok += len(hf_ids)
        n_bytefb += sum(sp.is_byte(i) for i in hf_ids)
    n = max(len(texts), 1)
    return {"hf_sp_agreement": agree / n, "roundtrip_exact": roundtrip / n,
            "byte_fallback_rate": n_bytefb / max(n_tok, 1), "n_texts": len(texts)}


def run(cfg) -> Path:
    set_seed(int(cfg.seed))
    tcfg = to_container(cfg.tokenizer)
    out_dir = ensure_dir(tcfg["output_dir"])
    save_run_metadata(cfg, out_dir, "kural_tokenizer")
    processed = Path(tcfg["processed_dir"])
    work = ensure_dir(out_dir / "_work")
    sample_cfg = tcfg.get("sample") or {}
    sample_file = work / "sample.txt"
    report = {"sample": write_sample(
        processed, dict(sample_cfg["category_weights"]), int(float(sample_cfg["total_chars"])),
        sample_file, int(cfg.seed), int(sample_cfg.get("max_line_chars", 4096)))}

    sp_model = train_sentencepiece(sample_file, work / "sp", tcfg)
    tok = export_hf(sp_model, out_dir, tcfg.get("chat_template") or DEFAULT_CHAT_TEMPLATE,
                    int(tcfg.get("model_max_length", 32768)))

    val_texts: list[str] = []
    for cat in available_categories(processed):
        for i, doc in enumerate(iter_category_docs(processed, cat, int(cfg.seed), split="eval")):
            if i >= 50:
                break
            val_texts.append(doc[:2000])
    report["validation"] = validate(tok, sp_model, val_texts)
    report["vocab_size"] = len(tok)
    log.info("Validation: %s", report["validation"])
    if report["validation"]["hf_sp_agreement"] < 0.99:
        log.warning("HF and SentencePiece encodings disagree on >1%% of texts; inspect before use")
    (out_dir / "tokenizer_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not tcfg.get("keep_sample", False):
        sample_file.unlink(missing_ok=True)
    log.info("Tokenizer saved to %s (vocab=%d)", out_dir, len(tok))
    return out_dir


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Train a Tamil SentencePiece tokenizer", argv))


if __name__ == "__main__":
    main()
