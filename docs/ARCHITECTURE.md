# Kural-270M — architecture and design decisions

Goal: adapt **Gemma 3 270M** heavily toward Tamil (formal, spoken, Tanglish,
Tamil–English code-switching) through continued pretraining, while keeping
basic English ability.

```
            ┌──────────────────────────── data/ ────────────────────────────┐
 HF / local │ sources ─► normalize ─► line filter ─► langid ─► route ─►     │
  corpora   │ quality ─► exact+MinHash dedup ─► train/eval split ─► shards  │
            │                                  └► synthetic Tanglish/spoken │
            └───────────────┬───────────────────────────────────────────────┘
                            │ processed/<category>/{train,eval}-*.jsonl.gz
        ┌───────────────────┴─────────────┐
        ▼                                 ▼
  tokenizer/train_tokenizer         data/packing (per training run)
  (SentencePiece BPE, Gemma ids)    tokenize ─► token-weighted mixture ─► pack
        │                                 │ packed/<run>/{train,eval_*}, manifest.json
        ▼                                 │
  tokenizer/adapt  (extend Gemma)         │
        │ extension_manifest.json         │
        ▼                                 ▼
  training/model_factory ─► training/pretrain (stage 1: embeddings, stage 2: full)
                                   │ checkpoints/<run>/final
                                   ▼
                            training/sft (assistant-only loss)
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
      evaluation/run_eval   inference/quantize    inference/generate
      (fertility, BPB,      (int8 / NF4 / GGUF    (prompt, chat,
       MC benchmarks)        + ΔBPB check)         interactive)
```

## 1. Configuration and reproducibility

* One YAML per stage, composed with `defaults:` (parents merged first), then
  CLI `key=value` overrides (OmegaConf). All paths derive from
  `paths.workspace` ← `$KURAL_WORKSPACE`; the base model from
  `$KURAL_BASE_MODEL`. No credentials in config: `HF_TOKEN`, `WANDB_API_KEY`
  are read by their libraries from the environment.
* Every stage writes `<stage>_config.yaml` (fully resolved) and
  `<stage>_metadata.json` (git commit + dirty flag, argv, library versions) next
  to its outputs.
* Seeds: global `seed` drives split assignment (hash-based, so a document's
  train/eval membership is stable across re-runs and source reorderings),
  tokenizer sampling, mixture selection, block shuffling and the Trainer.
* The packed-data manifest stores the tokenizer fingerprint; training refuses
  to start if the tokenizer vocabulary does not match.

## 2. Data pipeline (`data/`)

| step | module | notes |
|---|---|---|
| read | `sources.py` | HF (streaming), JSONL, text, Parquet; row filters, `max_docs` |
| normalize | `normalize.py` | HTML entities, zero-width chars (ZWJ/ZWNJ/BOM/soft hyphen), control chars, **NFC** (composes split two-part vowel signs ொ/ோ/ௌ), repeated vowel signs / virama, ஶ்ரீ→ஸ்ரீ, quotes/dashes, URL removal, e-mail/phone masking, whitespace |
| line filter | `quality.py` | boilerplate lines (English + Tamil patterns), `{}` lines, repeated lines |
| language ID | `langid.py` | script ratios + romanized-Tamil lexicon + suffix heuristics → `ta`, `code_mixed`, `tanglish`, `en`, `other`; optional fastText confirmation |
| routing | `pipeline.py` | doc goes to the source's category if it matches `lang_hint`, else to `code_mixed` / `tanglish` / `en` / `category_by_lang` |
| quality | `quality.py` | Gopher/C4-style statistics with **per-profile thresholds** (Tamil words are long; poetry repeats; comments are short) |
| dedup | `dedup.py` | exact (canonical xxhash64) + MinHash-LSH (word 5-gram shingles, 10 bands × 12 rows, Jaccard-verified ≥ 0.8), global across sources |
| augment | `tanglish.py` | synthetic **spoken Tamil** (formal→colloquial verb/pronoun endings) and **Tanglish** (contextual colloquial romanization, spelling-variant noise), written to separate `*_synthetic` categories |

Rejections are counted per source and reason in `stats.json`, so threshold
changes can be audited.

Scale notes: normalization/LID/quality/MinHash run in `num_workers`
processes; the dedup index is in-memory in the main process (fine to tens of
millions of docs). For larger corpora run per-source and add a cross-shard
dedup pass (the `Deduplicator` interface is small).

## 3. Tokenizer (`tokenizer/`)

Gemma 3's tokenizer is already multilingual (262k BPE pieces): on Tamil
Wikipedia it produces ~2–2.5 tokens per word. But its embedding matrix is
**~170M of the model's ~270M parameters**, so vocabulary decisions dominate
the parameter budget. Two strategies are implemented and can be compared on
the same data:

**`extend` (default).** Train a Tamil-heavy BPE (32k), take every piece made
purely of Tamil script that Gemma lacks, and add it to Gemma's SentencePiece
model:

* Gemma 3 has ~6.2k `<unusedN>` placeholder slots; new pieces fill them first,
  so the first ~6k Tamil pieces cost **no extra parameters**. The rest are
  appended (+640 params each).
* `score_policy: tamil_first` gives all Tamil-only pieces (new *and* already
  in Gemma) scores in the Tamil BPE's merge order, above Gemma's merges. Tamil
  then segments like the Tamil tokenizer; everything else (English, code,
  digits, other scripts) encodes **byte-identically** to Gemma — verified in
  `extension_manifest.json` (`identical_encoding_rate`).
* Taking the top-N pieces by BPE rank preserves merge closure (parents always
  rank higher), so every added piece is reachable.
* Output: `tokenizer.model` (for llama.cpp/GGUF) + `GemmaTokenizerFast`,
  identical special-token ids and chat markers.

On the smoke corpus (16k Tamil BPE, 6,012 pieces, all in reused slots) this
gives 18–19% fewer tokens on Tamil and spoken Tamil with zero new parameters.

**`replace`.** A 64k bilingual BPE replaces the vocabulary. The model shrinks
to ~140M parameters (embedding 41M), and every row is initialized from Gemma
(exact-piece copy, else mean of Gemma sub-token embeddings). Cheaper to train
and run, but English quality depends entirely on the new vocabulary and the
initial loss is much higher. Use it as an ablation.

Both tokenizers follow Gemma conventions (`<pad>=0 <eos>=1 <bos>=2 <unk>=3`,
byte fallback, split digits, no dummy prefix, identity normalization since the
data is already NFC).

**New-row init** (`embedding_init.py`): mean of the base embeddings of the
Gemma tokens the new piece used to split into; tied `lm_head` follows
automatically. This keeps the initial loss close to the base model's.

## 4. Continued pretraining (`training/`)

* **Mixture** (`data/packing.py`): weights are shares of **training tokens**.
  The budget is `data.token_budget` or, by default, one pass over the anchor
  (largest-weight) category. Small categories are up-sampled up to
  `max_epochs` (synthetic data: 1 epoch; English: 1 epoch); capped categories
  are logged and the achieved shares written to `manifest.json`.
* **English replay** (~17% of tokens) is the main defence against forgetting.
* **Two stages** (recommended for `extend`/`replace`):
  1. embeddings only (`trainable: embeddings`, LR 1e-3, ~1B tokens) — new rows
     settle without disturbing the transformer;
  2. full model (LR 2e-4, cosine to 10%, warmup 1%, wd 0.1, β₂ 0.95, clip 1.0).
* **Precision**: `precision: auto` → bf16 autocast with fp32 master weights on
  Ampere+; fp16 on older GPUs; fp32 on CPU.
* **Checkpoint/resume**: `save_steps` checkpoints contain model, optimizer,
  scheduler, RNG and data position; `resume: auto` restarts from the newest
  one (verified in the smoke run by killing and restarting at step 50).
* **Monitoring**: loss, grad-norm, LR, tokens seen, and per-language eval
  loss/perplexity (`eval_ta_ppl`, `eval_en_ppl`, `eval_tanglish_ppl`, …) to W&B
  (`online`/`offline`/`disabled`), plus greedy sample generations as a W&B table.
* Distributed: `torchrun`/`accelerate launch` work unchanged (HF Trainer / DDP).

## 5. SFT (`training/sft.py`, `training/chat_data.py`)

Alpaca / prompt-response / OpenAI-messages / ShareGPT inputs are normalized
into Gemma chat turns (`<start_of_turn>user … <end_of_turn>`); loss is applied
only to assistant content plus its `<end_of_turn>`. The rendering is
unit-tested to match the tokenizer's chat template, so training and inference
prompts are identical. Exact-duplicate conversations are dropped; eval split is
hash-based. Small English instruction replay is on by default.

## 6. Evaluation (`evaluation/`)

* **Tokenizers** on identical text: fertility (tokens/word), bytes/token,
  tokens/grapheme, continued-word ratio, byte-fallback rate, round-trip.
* **Language modelling**: bits-per-byte (primary; comparable across
  tokenizers), bits-per-char, word-level perplexity, token perplexity (same
  tokenizer only), sliding-window scoring. Token PPL of models with different
  tokenizers is **not** comparable — the report says so.
* **Benchmarks** (zero-shot log-likelihood, `acc` and byte-normalized
  `acc_norm`): IndicCOPA-ta, IndicSentiment-ta, and a hand-written 22-item
  sanity set covering Tamil facts, grammar (tense/person agreement), spoken
  Tamil, Tanglish and English.
* **Samples**: fixed prompts, greedy decoding, recorded in the report.

## 7. Inference & quantization (`inference/`)

* `dynamic_int8` (CPU, no deps), `bnb_8bit` / `bnb_4bit` (CUDA), `gguf`
  (llama.cpp `convert_hf_to_gguf.py` + `llama-quantize`, e.g. Q8_0/Q4_K_M).
  Each torch-loadable result is re-scored for ΔBPB against the fp32 model.
* `generate.py`: single prompts, batch prompts, or interactive chat with
  history, sampling parameters from config.

## 8. Known limitations / next steps

* Tamil LID is heuristic; add a fastText model (`KURAL_FASTTEXT_LID`) for
  web-scale runs and spot-check `stats.json` rejections per source.
* The Tanglish lexicon and colloquialization rules are hand-written; synthetic
  data is capped (1 epoch, small weight) and kept in separate categories so its
  effect can be ablated.
* Public spoken-Tamil and Tanglish corpora are scarce — the `local_*` sources
  are the hook for licensed transcripts, subtitles and social data.
* In-memory dedup and non-streaming packing are simple by design; move to
  sharded dedup and streaming packing above ~100B tokens.
