# kural-270M
அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு.

A Tamil-specific small language model: **continued pretraining of Gemma 3 270M**
on cleaned Tamil (formal, spoken, Tanglish, Tamil–English code-switching) with
English replay, followed by Tamil instruction tuning.

The design rationale is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository layout

```
configs/      YAML configs (composable via `defaults:`, overridable as key=value)
  base.yaml     workspace paths (from $KURAL_WORKSPACE), base model, W&B
  data/         corpus pipeline (full + smoke)
  tokenizer/    SentencePiece training + Gemma vocabulary extension
  model/        gemma3_270m_extend | gemma3_270m_replace | scratch_10m
  train/        mixtures, 10M experimental, 270M CPT (single / two-stage), 270M smoke
  sft/ eval/ inference/
common/       config loader, logging, seeding, JSONL shards, Tamil script helpers
data/         sources, normalization, language ID, quality, dedup, Tanglish synth, pipeline, packing
tokenizer/    train_tokenizer.py, adapt.py (extend Gemma), embedding_init.py
training/     model_factory.py, pretrain.py, sft.py, chat_data.py, common.py
evaluation/   tokenizer_compare.py, perplexity.py (bits/byte), benchmarks.py, run_eval.py
inference/    quantize.py (int8 / NF4 / GGUF), generate.py
scripts/      run_experimental.py (end-to-end smoke), run_full_270m.sh
tests/        unit tests (no network needed)
```

All generated artifacts go to `$KURAL_WORKSPACE` (default `./workspace`, git-ignored):
`processed/`, `tokenizers/`, `packed/`, `checkpoints/`, `reports/`, `exports/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # + bitsandbytes / torchao for GPU quantization
cp .env.example .env                                 # fill in, then export the variables
```

Environment variables (nothing is hard-coded):

| variable | purpose |
|---|---|
| `KURAL_WORKSPACE` | root for all outputs |
| `KURAL_BASE_MODEL` | base checkpoint (default `google/gemma-3-270m`, gated → accept license + `HF_TOKEN`; `unsloth/gemma-3-270m` is an ungated mirror) |
| `HF_TOKEN` | Hugging Face access (gated models/datasets) |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE` | experiment tracking (`offline` needs no key) |
| `KURAL_FASTTEXT_LID` | optional fastText `lid.176.bin` for extra language-ID confirmation |
| `LLAMA_CPP_DIR` | optional llama.cpp checkout for GGUF export |

## 1. Verify everything with the experimental run (~25 min on a laptop CPU)

```bash
export KURAL_BASE_MODEL=unsloth/gemma-3-270m   # or google/gemma-3-270m with HF_TOKEN
python scripts/run_experimental.py             # add --with-base-model to also train 6 steps of real Gemma 270M
python -m pytest                               # unit tests
```

This runs, on ~7.5k Tamil/English Wikipedia docs + the bundled Tanglish/spoken seed set:

1. **data** – cleaning, LID, dedup, split, synthetic Tanglish/spoken Tamil → `processed/smoke`
2. **tokenizer** – 16k Tamil BPE (HF ↔ SentencePiece agreement and round-trip checked)
3. **extend** – Gemma 3 tokenizer + Tamil pieces (reuses `<unused>` slots; English unchanged)
4. **pretrain** – ~10M-param Gemma 3 architecture from scratch, 200 steps, per-language eval ppl, checkpoints (try killing it and re-running: it resumes)
5. **sft** – chat-format SFT on bundled examples with assistant-only loss
6. **quantize** – dynamic int8 + ΔBPB check
7. **eval** – tokenizer comparison (Gemma vs Tamil vs extended), bits-per-byte, benchmarks, samples → `reports/eval_smoke/report.md`
8. **generate** – chat generation from the quantized model

Any step can be run alone, e.g. `python scripts/run_experimental.py --only eval`, or
directly: `python -m training.pretrain --config configs/train/experimental_10m.yaml train.args.max_steps=50`.

## 2. Full 270M run

```bash
export KURAL_WORKSPACE=/data/kural HF_TOKEN=... WANDB_API_KEY=...
NGPUS=8 bash scripts/run_full_270m.sh
```

Steps: `data` (configs/data/pipeline.yaml — review sources and licenses first) →
`tokenizer` (32k Tamil BPE) → `extend` → `pack` (10B-token mixture, seq 2048) →
`stage1` (embeddings only) → `stage2` (full CPT) → `sft` → `eval` → `quantize`.
Each is resumable. Budgets, mixture weights, LR schedule, batch size etc. live in
`configs/train/cpt_270m*.yaml`; the vocabulary-replacement ablation is
`configs/model/gemma3_270m_replace.yaml` + `configs/tokenizer/bilingual_bpe_64k.yaml`.

Before launching, check on the smoke outputs:

* `processed/*/stats.json` – rejection reasons per source look sane
* `tokenizers/*/extension_manifest.json` – Tamil token reduction, `identical_encoding_rate == 1.0` for English
* `packed/*/manifest.json` – achieved mixture shares vs targets
* W&B (or `./wandb` offline runs) – loss decreasing, `eval_*_ppl` per language

## Data sources

Every dataset the project reads, where it comes from, and where it's used. **Status** says
what the pipeline actually touched:

- **used (smoke)**: read in the experimental run.
- **configured**: set up for the full run but not read yet. Tamil SQuAD, Tamil Alpaca and Tamil Alpaca-Orca have been load-tested through the pipeline code; the others were only confirmed to exist on the Hub.
- **gated**: needs access approval on the Hub plus `HF_TOKEN`.
- **disabled**: present in the config but turned off.

Check each license before using a model trained on the data.

### Pretraining corpus (`configs/data/pipeline.yaml`, `pipeline_smoke.yaml`)

| Source | Link | Category | License | Status | Notes |
|---|---|---|---|---|---|
| Tamil Wikipedia (2023-11-01 dump) | [wikimedia/wikipedia · 20231101.ta](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.ta/train) | `ta_wiki` | CC BY-SA 3.0 / GFDL | used (smoke: first 6,000 articles) | Encyclopedic, formal Tamil |
| English Wikipedia (2023-11-01 dump) | [wikimedia/wikipedia · 20231101.en](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.en/train) | `en` | CC BY-SA 3.0 / GFDL | used (smoke: first 1,500 articles) | English replay in the smoke run |
| AI4Bharat Sangraha (verified, Tamil) | [ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) | `ta_web` | CC BY 4.0 | configured | Cleaned web + PDF text |
| FineWeb-2 (Tamil) | [HuggingFaceFW/fineweb-2 · tam_Taml](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | `ta_web` | ODC-By (CommonCrawl terms apply) | configured | Large filtered web crawl |
| CulturaX (Tamil) | [uonlp/CulturaX · ta](https://huggingface.co/datasets/uonlp/CulturaX) | `ta_web` | see dataset card (mC4 + OSCAR terms) | **gated** | Read directly from `ta/*.parquet` (the repo's loading script isn't supported by `datasets`≥4). Skipped with a logged error if you lack access. |
| Tamil SQuAD 2.0 passages | [RajeevanL/tamil_squad-2.0](https://huggingface.co/datasets/RajeevanL/tamil_squad-2.0) | `ta_translated` | not stated on card | configured | ~17k unique passages translated from English Wikipedia; `சூழல்:` label stripped; small weight (1%) |
| FineWeb-Edu (English sample) | [HuggingFaceFW/fineweb-edu · sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `en` | ODC-By | configured | English replay (~17% of tokens), capped at 3M docs |
| Dravidian code-mix (Tamil) | [community-datasets/offenseval_dravidian · tamil](https://huggingface.co/datasets/community-datasets/offenseval_dravidian) | `tanglish` | CC BY 4.0 | disabled | Tanglish YouTube comments (short, offensive-language labels) |
| Local files | `$KURAL_WORKSPACE/raw/{literature,news,spoken,tanglish}/` | `ta_literature`, `ta_news`, `ta_spoken`, `tanglish` | yours | disabled | Hooks for licensed text you add yourself |
| Bundled Tanglish / spoken seed | [`data/samples/tanglish_spoken_sample.jsonl`](data/samples/tanglish_spoken_sample.jsonl) | `tanglish`, `ta_spoken` | Apache-2.0 (this repo) | used (smoke) | 30 hand-written lines; for pipeline testing only |
| Synthetic Tanglish / spoken Tamil | generated by [`data/tanglish.py`](data/tanglish.py) from Tamil docs | `tanglish_synthetic`, `ta_spoken_synthetic` | derived from source docs | used (smoke) | Rule-based romanization / colloquialization; capped at 1 epoch |

### Instruction tuning (`configs/sft/sft_270m.yaml`)

| Source | Link | Format | Rows | License | Status | Notes |
|---|---|---|---|---|---|---|
| Tamil Alpaca | [abhinand/tamil-alpaca](https://huggingface.co/datasets/abhinand/tamil-alpaca) | alpaca | 51,876 | GPL-3.0 | configured | Translated Alpaca |
| Tamil Alpaca-Orca | [abhinand/tamil-alpaca-orca](https://huggingface.co/datasets/abhinand/tamil-alpaca-orca) | alpaca | 145,181 | GPL-3.0 | configured | Includes all Tamil Alpaca rows plus ~93k Orca (FLAN/CoT/T0/NIv2); exact duplicates are dropped |
| Tamil SQuAD 2.0 | [RajeevanL/tamil_squad-2.0](https://huggingface.co/datasets/RajeevanL/tamil_squad-2.0) | qa | 66,277 (train) | not stated on card | configured | Reading comprehension; rows with no answer are dropped; validation/test splits unused |
| Aya dataset (Tamil) | [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) | prompt_response | filtered `language == Tamil` | Apache-2.0 | configured | Human-written prompts/answers |
| Alpaca-cleaned (English) | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | alpaca | first 10,000 | CC BY 4.0 | configured | English replay |
| Local chats | `$KURAL_WORKSPACE/raw/sft/*.jsonl` | messages | – | yours | disabled | Your own conversations |
| Bundled SFT sample | [`data/samples/sft_sample.jsonl`](data/samples/sft_sample.jsonl) | alpaca | 16 | Apache-2.0 (this repo) | used (smoke) | Hand-written; smoke test only |

### Evaluation (`configs/eval/`)

| Source | Link | Use | License |
|---|---|---|---|
| Held-out split of the pretraining corpus | `$KURAL_WORKSPACE/processed/*/*/eval-*.jsonl` | bits-per-byte, tokenizer fertility | as above |
| IndicCOPA (Tamil test, 500) | [ai4bharat/IndicCOPA](https://huggingface.co/datasets/ai4bharat/IndicCOPA) | commonsense reasoning (2-choice) | CC BY 4.0 |
| IndicSentiment (Tamil test, 1,000) | [ai4bharat/IndicSentiment](https://huggingface.co/datasets/ai4bharat/IndicSentiment) | sentiment (2-choice) | see dataset card |
| Tamil cloze sanity set (22 items) | [`evaluation/resources/tamil_cloze.jsonl`](evaluation/resources/tamil_cloze.jsonl) | facts, grammar, spoken, Tanglish, English | Apache-2.0 (this repo) |

### Base model

| Model | Link | License |
|---|---|---|
| Gemma 3 270M | [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) (gated), mirror [unsloth/gemma-3-270m](https://huggingface.co/unsloth/gemma-3-270m) | Gemma Terms of Use |

`data/resources/` only holds word lists used for filtering (Tanglish lexicon, stopwords,
boilerplate patterns). It contains no training text.

## 3. CPU-only training (laptop recipe)

Measured on a 4-core laptop CPU: full training of Gemma 3 270M runs at ~40 tokens/s (bf16 is far
slower on CPUs without native bf16). Continued pretraining at useful scale is therefore GPU-only
(10B tokens ≈ 8 years on CPU). What fits on a CPU is **Tamil instruction tuning** of Google's
instruction-tuned checkpoint, ~2.2k conversations in ~2.5 hours:

```bash
python -m training.sft --config configs/sft/sft_270m_cpu.yaml           # train (checkpoints every 20 steps; re-run resumes)
python -m evaluation.run_eval --config configs/eval/eval_cpu.yaml       # before/after metrics
python -m inference.quantize --config configs/inference/quantize.yaml \
  quantize.model_path=workspace/checkpoints/kural-270m-cpu-sft/final \
  quantize.output_dir=workspace/exports/kural-270m-cpu-sft quantize.methods=[dynamic_int8] \
  quantize.eval_texts.processed_dir=workspace/processed/sample
python -m inference.server --config configs/inference/eval_server_cpu.yaml   # test in the browser
```

Two SFT speed-ups make this feasible (2.3× faster, a third less memory): the 168M-parameter tied
embedding is frozen (`trainable: no_embeddings`, vocabulary unchanged) and the 262k-way output
layer is computed only at answer tokens (`sft.label_only_logits`). The original Gemma tokenizer
is kept on purpose: new Tamil token embeddings need far more training data than a CPU can process.
Close other heavy apps (browsers, Teams) while training — they compete for the same cores.

## Evaluating the chatbot (web UI)

```bash
python -m inference.server --config configs/inference/eval_server.yaml        # trained Kural models + baselines
python -m inference.server --config configs/inference/eval_server_smoke.yaml  # what exists after the smoke runs
# open http://127.0.0.1:7860
```

The page (`inference/web/eval_chat.html`, served by `inference/server.py`) has four tabs
(Blind compare, Chat, Test set, Results). **Test set** runs all 26 evaluation prompts on the
models you tick and shows the answers side by side next to the reference; ✓ / ✗ marks are saved
as correctness ratings. The other three:

- **Blind compare**: pick a prompt from the 26-prompt Tamil evaluation set
  (`evaluation/resources/chat_eval_prompts.jsonl`: facts, explanation, writing, grammar, spoken Tamil,
  Tanglish, code-mixed, translation, reasoning, culture, English, safety) or type your own.
  Two randomly chosen models answer side by side with their names hidden. Vote A / tie / B / both bad
  (keys 1–4), optionally score each answer 1–5 for correctness, natural Tamil and helpfulness, then
  the names are revealed.
- **Chat**: free multi-turn chat with any model, and rate individual answers.
- **Results**: win rate per model, win rate by prompt category, head-to-head counts, mean scores,
  and a JSONL download.

Ratings are appended to `$KURAL_WORKSPACE/reports/human_eval/ratings.jsonl`. Summarize them offline with
`python -m evaluation.human_eval --ratings <file> --out report.md`. Models listed in the config that
don't exist yet (e.g. before training) show as unavailable. Generation settings are shared by both
sides of a comparison; use temperature 0 for reproducible runs. The server has no authentication, so
keep it on `127.0.0.1`.

For automatic metrics (bits-per-byte, IndicCOPA, IndicSentiment, cloze) use `python -m evaluation.run_eval`.

## Inference

```bash
python -m inference.generate --config configs/inference/generate.yaml inference.interactive=true
python -m inference.quantize --config configs/inference/quantize.yaml
```

## License

Apache-2.0 for this code. Gemma weights are subject to the Gemma Terms of Use;
each dataset keeps its own license.
