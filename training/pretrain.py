"""Continued causal-LM pretraining (and from-scratch experimental runs).

    python -m training.pretrain --config configs/train/experimental_10m.yaml
    torchrun --nproc_per_node 8 -m training.pretrain --config configs/train/cpt_270m.yaml
    accelerate launch -m training.pretrain --config configs/train/cpt_270m.yaml

* data: packed blocks from ``data.packing`` (built automatically if missing and
  ``data.auto_prepare`` is true)
* precision: ``train.precision: auto|bf16|fp16|fp32`` (auto → bf16 on capable GPUs)
* gradient accumulation / checkpointing / optimizer / schedule: ``train.args``
  (passed verbatim to ``transformers.TrainingArguments``)
* resume: ``train.resume: auto`` picks up the newest checkpoint in the output dir;
  the Trainer restores optimizer, scheduler, RNG and data position
* eval: one loss/perplexity per ``data.eval_sets`` entry (``eval_ta_ppl``, ``eval_en_ppl``…)
"""

from __future__ import annotations

import json
from pathlib import Path

from common.config import parse_config, save_run_metadata, to_container
from common.utils import get_logger, set_seed, setup_logging
from training.common import (KuralTrainer, SampleGenerationCallback, build_training_args, resolve_resume,
                             setup_wandb)
from training.model_factory import apply_trainable, build_model_and_tokenizer

log = get_logger("training.pretrain")


def load_packed(cfg):
    from datasets import load_from_disk

    packed = Path(cfg.data.packed_dir)
    if not (packed / "manifest.json").exists():
        if not cfg.data.get("auto_prepare", True):
            raise FileNotFoundError(f"No packed data at {packed}; run python -m data.packing first")
        from data.packing import run as pack

        log.info("Packed data missing; building it now")
        pack(cfg)
    manifest = json.loads((packed / "manifest.json").read_text(encoding="utf-8"))
    train = load_from_disk(str(packed / "train"))
    evals = {name: load_from_disk(str(packed / f"eval_{name}")) for name in manifest.get("eval", {})}
    return train, evals, manifest


def run(cfg) -> Path:
    from transformers import DataCollatorForLanguageModeling

    set_seed(int(cfg.seed))
    run_name = str(cfg.run_name)
    output_dir = str(cfg.train.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, output_dir, "kural_train")
    report_to = setup_wandb(cfg, run_name)

    model, tok, info = build_model_and_tokenizer(cfg)
    info.update(apply_trainable(model, str(cfg.train.get("trainable", "all"))))
    train_ds, eval_sets, manifest = load_packed(cfg)
    if manifest["tokenizer_vocab_size"] != len(tok):
        raise ValueError(f"Packed data tokenizer vocab ({manifest['tokenizer_vocab_size']}) != model tokenizer "
                         f"({len(tok)}); rebuild with data.overwrite_packed=true")
    args = build_training_args(cfg, output_dir, report_to, run_name)
    if args.gradient_checkpointing:
        model.config.use_cache = False
    world = max(args.world_size, 1)
    tokens_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * world * manifest["seq_len"]
    log.info("Train blocks=%d, seq_len=%d, tokens/step=%d, max_steps=%s", len(train_ds), manifest["seq_len"],
             tokens_per_step, args.max_steps)

    callbacks = []
    gen = to_container(cfg.get("sample_generation") or {})
    if gen.get("prompts"):
        callbacks.append(SampleGenerationCallback(tok, list(gen["prompts"]), int(gen.get("max_new_tokens", 48))))

    trainer = KuralTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_sets or None,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        processing_class=tok,
        callbacks=callbacks,
        tokens_per_step=tokens_per_step,
    )
    resume = resolve_resume(cfg.train.get("resume", "auto"), output_dir)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    if eval_sets:
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)
    (final_dir / "kural_model_info.json").write_text(
        json.dumps({**info, "run_name": run_name, "packed_manifest": str(Path(cfg.data.packed_dir) / "manifest.json")},
                   indent=2), encoding="utf-8")
    log.info("Saved final model to %s", final_dir)
    return final_dir


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Continued pretraining for Kural (Tamil Gemma 3)", argv))


if __name__ == "__main__":
    main()
