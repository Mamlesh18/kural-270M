"""Supervised fine-tuning on Tamil instruction / chat data.

    python -m training.sft --config configs/sft/sft_270m.yaml

Loads the continued-pretraining checkpoint (``model.source: checkpoint``), builds
Gemma-format conversations with assistant-only loss, holds out a deterministic
eval split, and trains with the same Trainer plumbing as pretraining (precision,
accumulation, checkpoint/resume, W&B, sample generations in chat mode).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from common.config import parse_config, save_run_metadata, to_container
from common.utils import get_logger, set_seed, setup_logging, stable_fraction, text_hash
from data.normalize import TextNormalizer
from training.chat_data import ChatCollator, encode_conversation, iter_conversations
from training.common import KuralTrainer, SampleGenerationCallback, build_training_args, resolve_resume, setup_wandb
from training.model_factory import apply_trainable, build_model_and_tokenizer

log = get_logger("training.sft")


def build_datasets(cfg, tok):
    from datasets import Dataset

    s = to_container(cfg.sft)
    normalizer = TextNormalizer(s.get("normalize")) if s.get("normalize", True) is not False else None
    max_len = int(s.get("max_seq_len", 1024))
    truncation = s.get("truncation", "drop")
    eval_fraction = float(s.get("eval_fraction", 0.02))
    repeat = int(s.get("repeat", 1))
    train, evals, seen = [], [], set()
    dropped = 0
    for conv in iter_conversations(s["sources"], normalizer, s.get("system_prompt")):
        key = text_hash(json.dumps(conv["messages"], ensure_ascii=False))
        if key in seen:  # exact duplicate conversations
            continue
        seen.add(key)
        ex = encode_conversation(tok, conv["messages"], max_len, truncation)
        if ex is None:
            dropped += 1
            continue
        (evals if stable_fraction(key, str(cfg.seed)) < eval_fraction else train).append(ex)
    if not train:
        raise ValueError("No SFT training examples (check sources / max_seq_len)")
    if not evals:
        evals = train[: max(1, len(train) // 20)]
        log.warning("Eval split empty; using %d training examples for eval", len(evals))
    train = train * repeat
    random.Random(int(cfg.seed)).shuffle(train)
    log.info("SFT examples: train=%d eval=%d dropped=%d (too long / empty)", len(train), len(evals), dropped)
    n_label = sum(sum(x != -100 for x in e["labels"]) for e in train)
    n_tok = sum(len(e["input_ids"]) for e in train)
    log.info("Trainable tokens: %d / %d (%.1f%%)", n_label, n_tok, 100 * n_label / max(n_tok, 1))
    return Dataset.from_list(train), Dataset.from_list(evals)


def run(cfg) -> Path:
    set_seed(int(cfg.seed))
    run_name = str(cfg.run_name)
    output_dir = str(cfg.train.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, output_dir, "kural_sft")
    report_to = setup_wandb(cfg, run_name)

    model, tok, info = build_model_and_tokenizer(cfg)
    info.update(apply_trainable(model, str(cfg.train.get("trainable", "all"))))
    train_ds, eval_ds = build_datasets(cfg, tok)
    args = build_training_args(cfg, output_dir, report_to, run_name)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    callbacks = []
    gen = to_container(cfg.get("sample_generation") or {})
    if gen.get("prompts"):
        callbacks.append(SampleGenerationCallback(tok, list(gen["prompts"]), int(gen.get("max_new_tokens", 64)),
                                                  chat=True))
    trainer = KuralTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=ChatCollator(tok.pad_token_id), processing_class=tok, callbacks=callbacks,
    )
    result = trainer.train(resume_from_checkpoint=resolve_resume(cfg.train.get("resume", "auto"), output_dir))
    trainer.save_metrics("train", result.metrics)
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    trainer.save_metrics("eval", trainer.evaluate())
    (final_dir / "kural_model_info.json").write_text(json.dumps({**info, "run_name": run_name}, indent=2),
                                                     encoding="utf-8")
    log.info("Saved SFT model to %s", final_dir)
    return final_dir


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Supervised fine-tuning on Tamil instruction data", argv))


if __name__ == "__main__":
    main()
