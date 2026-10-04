#!/usr/bin/env bash
# Full Kural 270M pipeline on a GPU node. Every step is resumable / idempotent:
# re-running skips packed data that already exists and resumes training from the
# newest checkpoint (train.resume=auto).
#
#   export KURAL_WORKSPACE=/data/kural HF_TOKEN=... WANDB_API_KEY=...
#   NGPUS=8 bash scripts/run_full_270m.sh            # all steps
#   NGPUS=8 bash scripts/run_full_270m.sh cpt sft    # selected steps
set -euo pipefail
cd "$(dirname "$0")/.."

NGPUS="${NGPUS:-1}"
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

launch() {  # launch <module> <config> [overrides...]
  local module="$1"; shift
  if [ "$NGPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node "$NGPUS" -m "$module" --config "$@"
  else
    "$PY" -m "$module" --config "$@"
  fi
}

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(data tokenizer extend pack stage1 stage2 sft eval quantize)

for step in "${STEPS[@]}"; do
  echo "=== $step"
  case "$step" in
    data)      "$PY" -m data.pipeline --config configs/data/pipeline.yaml ;;
    tokenizer) "$PY" -m tokenizer.train_tokenizer --config configs/tokenizer/tamil_bpe_32k.yaml ;;
    tokenizer_replace) "$PY" -m tokenizer.train_tokenizer --config configs/tokenizer/bilingual_bpe_64k.yaml ;;
    extend)    "$PY" -m tokenizer.adapt --config configs/tokenizer/extend_gemma.yaml ;;
    # Pack once on a single process (tokenization uses data.num_proc workers) before
    # launching distributed training, so ranks don't race to build it.
    pack)      "$PY" -m data.packing --config configs/train/cpt_270m.yaml ;;
    stage1)    launch training.pretrain configs/train/cpt_270m_stage1_embeddings.yaml ;;
    stage2)    launch training.pretrain configs/train/cpt_270m_stage2_full.yaml ;;
    cpt)       launch training.pretrain configs/train/cpt_270m.yaml ;;   # single-stage alternative
    sft)       launch training.sft configs/sft/sft_270m.yaml ;;
    eval)      "$PY" -m evaluation.run_eval --config configs/eval/eval_270m.yaml ;;
    quantize)  "$PY" -m inference.quantize --config configs/inference/quantize.yaml ;;
    *) echo "unknown step: $step" >&2; exit 2 ;;
  esac
done
