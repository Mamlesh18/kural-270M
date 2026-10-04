"""Parameter count of Gemma 3 270M under each tokenizer strategy (no training).

    python scripts/count_params.py --extended workspace/tokenizers/gemma3_tamil_extended \
                                   --replace workspace/tokenizers/bilingual_bpe_64k

Loads the base model, applies the same vocabulary adaptation as training
(``training.model_factory``: resize + embedding init) and reports total /
embedding / transformer parameters. Uses ``$KURAL_BASE_MODEL``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.utils import setup_logging  # noqa: E402
from tokenizer.embedding_init import resize_and_init  # noqa: E402
from training.model_factory import param_report  # noqa: E402


def main() -> None:
    setup_logging("WARNING")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.environ.get("KURAL_BASE_MODEL", "google/gemma-3-270m"))
    ap.add_argument("--extended", action="append", default=[], help="extended tokenizer dir (repeatable)")
    ap.add_argument("--replace", action="append", default=[], help="replacement tokenizer dir (repeatable)")
    ap.add_argument("--pad-to", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(args.base)
    rows = []

    def measure(label: str, tok_dir: str | None, mode: str) -> None:
        model = AutoModelForCausalLM.from_pretrained(args.base)
        vocab = len(base_tok)
        if tok_dir:
            tok = AutoTokenizer.from_pretrained(tok_dir)
            vocab = len(tok)
            ids = None
            if mode == "extend":
                ids = json.loads((Path(tok_dir) / "extension_manifest.json").read_text(encoding="utf-8"))["new_token_ids"]
            resize_and_init(model, base_tok, tok, ids, args.pad_to)
        r = param_report(model)
        rows.append((label, vocab, model.get_input_embeddings().weight.shape[0], r))

    measure("gemma3-270m (original)", None, "none")
    for d in args.extended:
        measure(f"extend: {Path(d).name}", d, "extend")
    for d in args.replace:
        measure(f"replace: {Path(d).name}", d, "replace")

    base_total = rows[0][3]["params_total"]
    print(f"\n{'model':44s} {'tokenizer':>9s} {'emb rows':>9s} {'total':>10s} {'embedding':>10s} "
          f"{'transformer':>11s} {'vs base':>14s}")
    for label, vocab, emb_rows, r in rows:
        delta = r["params_total"] - base_total
        print(f"{label:44s} {vocab:9,d} {emb_rows:9,d} {r['params_total'] / 1e6:9.2f}M "
              f"{r['params_embedding'] / 1e6:9.2f}M {r['params_non_embedding'] / 1e6:10.2f}M "
              f"{delta / 1e6:+8.2f}M ({100 * delta / base_total:+.1f}%)")


if __name__ == "__main__":
    main()
