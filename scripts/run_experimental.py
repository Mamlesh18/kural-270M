"""End-to-end smoke / experimental pipeline (cross-platform).

    python scripts/run_experimental.py                 # all steps
    python scripts/run_experimental.py --from pretrain # resume from a step
    python scripts/run_experimental.py --only eval
    python scripts/run_experimental.py --with-base-model   # also run the real Gemma 270M extend smoke

Steps (each is an independent module you can also run by hand):

  data       python -m data.pipeline         configs/data/pipeline_smoke.yaml
  tokenizer  python -m tokenizer.train_tokenizer configs/tokenizer/tamil_bpe_smoke.yaml
  extend     python -m tokenizer.adapt       configs/tokenizer/extend_gemma_smoke.yaml   (needs base tokenizer access)
  pretrain   python -m training.pretrain     configs/train/experimental_10m.yaml (packs data automatically)
  gemma      python -m training.pretrain     configs/train/smoke_gemma270m_extend.yaml   (--with-base-model)
  sft        python -m training.sft          configs/sft/sft_smoke.yaml
  quantize   python -m inference.quantize    configs/inference/quantize_smoke.yaml
  eval       python -m evaluation.run_eval   configs/eval/eval_smoke.yaml
  generate   python -m inference.generate    configs/inference/generate_smoke.yaml

Extra ``key=value`` arguments after ``--`` are forwarded to every step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STEPS = [
    ("data", "data.pipeline", "configs/data/pipeline_smoke.yaml"),
    ("tokenizer", "tokenizer.train_tokenizer", "configs/tokenizer/tamil_bpe_smoke.yaml"),
    ("extend", "tokenizer.adapt", "configs/tokenizer/extend_gemma_smoke.yaml"),
    ("pretrain", "training.pretrain", "configs/train/experimental_10m.yaml"),
    ("gemma", "training.pretrain", "configs/train/smoke_gemma270m_extend.yaml"),
    ("sft", "training.sft", "configs/sft/sft_smoke.yaml"),
    ("quantize", "inference.quantize", "configs/inference/quantize_smoke.yaml"),
    ("eval", "evaluation.run_eval", "configs/eval/eval_smoke.yaml"),
    ("generate", "inference.generate", "configs/inference/generate_smoke.yaml"),
]
OPTIONAL = {"gemma"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [s[0] for s in STEPS]
    ap.add_argument("--from", dest="start", choices=names, default=names[0])
    ap.add_argument("--only", choices=names)
    ap.add_argument("--with-base-model", action="store_true",
                    help="also run a few steps on the real Gemma 3 270M (downloads ~0.5 GB)")
    ap.add_argument("--skip-extend", action="store_true", help="skip steps that need the Gemma tokenizer")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    selected = [s for s in STEPS if s[0] == args.only] if args.only else STEPS[names.index(args.start):]
    for name, module, config in selected:
        if name in OPTIONAL and not (args.with_base_model or args.only == name):
            continue
        if name == "extend" and args.skip_extend:
            continue
        cmd = [sys.executable, "-m", module, "--config", config, *args.overrides]
        print(f"\n=== [{name}] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT, env=env)
        print(f"=== [{name}] exit={rc} ({time.time() - t0:.0f}s)", flush=True)
        if rc != 0:
            print(f"Step {name} failed; fix and rerun with --from {name}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
