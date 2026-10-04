"""Tokenizer extension, embedding init, SFT masking and mixture planning on tiny synthetic data.

Everything is trained locally in a temp dir (no network access needed).
"""

import random

import pytest

sentencepiece = pytest.importorskip("sentencepiece")

from data.packing import _select, compute_plan  # noqa: E402
from tokenizer.adapt import compare_encodings, extend_sp_model  # noqa: E402
from tokenizer.train_tokenizer import export_hf, train_sentencepiece  # noqa: E402
from training.chat_data import IGNORE, ChatCollator, encode_conversation, row_to_messages  # noqa: E402

EN = ["the cat sat on the mat", "a quick brown fox jumps over the lazy dog", "language models learn from text",
      "we train small models on a laptop", "english text must keep its tokenization"]
TA = ["தமிழ்நாட்டின் தலைநகரம் சென்னை", "திருக்குறள் ஒரு சிறந்த நூல்", "நான் நாளை ஊருக்குப் போகிறேன்",
      "தமிழ் மொழி மிகவும் பழமையானது", "அவர்கள் பள்ளிக்குச் சென்றார்கள்"]
SP_ARGS = {"model_type": "bpe", "character_coverage": 1.0, "input_sentence_size": 100000, "num_threads": 1,
           "user_defined_symbols": ["<start_of_turn>", "<end_of_turn>"], "minloglevel": 2,
           "hard_vocab_limit": False}


@pytest.fixture(scope="module")
def tokenizers(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tok")
    rng = random.Random(0)
    (tmp / "en.txt").write_text("\n".join(rng.choice(EN) for _ in range(3000)), encoding="utf-8")
    (tmp / "ta.txt").write_text("\n".join(rng.choice(TA) for _ in range(3000)), encoding="utf-8")
    base_sp = train_sentencepiece(tmp / "en.txt", tmp / "base", {**SP_ARGS, "vocab_size": 300})
    ta_sp = train_sentencepiece(tmp / "ta.txt", tmp / "ta", {**SP_ARGS, "vocab_size": 300})
    base_dir, ta_dir = tmp / "base_hf", tmp / "ta_hf"
    base_tok = export_hf(base_sp, base_dir)
    export_hf(ta_sp, ta_dir)
    return tmp, base_sp, ta_sp, base_tok


def test_hf_export_roundtrip(tokenizers):
    _, _, _, base_tok = tokenizers
    for s in EN:
        assert base_tok.decode(base_tok(s, add_special_tokens=False)["input_ids"]) == s
    assert base_tok.bos_token_id == 2 and base_tok.eos_token_id == 1 and base_tok.pad_token_id == 0


@pytest.mark.parametrize("policy", ["tamil_first", "append"])
def test_extend_keeps_english_and_compresses_tamil(tokenizers, policy):
    tmp, base_sp, ta_sp, base_tok = tokenizers
    merged, manifest = extend_sp_model(base_sp, ta_sp, reuse_unused=False, score_policy=policy)
    assert manifest["num_new_pieces"] > 0
    path = tmp / f"merged_{policy}.model"
    path.write_bytes(merged.SerializeToString())
    new_tok = export_hf(path, tmp / f"merged_{policy}_hf")
    rep = compare_encodings(base_tok, new_tok, {"en": EN, "ta": TA})
    assert rep["en"]["identical_encoding_rate"] == 1.0
    assert rep["ta"]["token_reduction"] > 0.3
    assert rep["ta"]["roundtrip_exact"] == 1.0
    # Base ids are untouched.
    for piece, idx in base_tok.get_vocab().items():
        assert new_tok.convert_tokens_to_ids(piece) == idx


def test_embedding_init_for_new_tokens(tokenizers):
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    from tokenizer.embedding_init import resize_and_init

    tmp, base_sp, ta_sp, base_tok = tokenizers
    merged, manifest = extend_sp_model(base_sp, ta_sp, reuse_unused=False)
    path = tmp / "merged_emb.model"
    path.write_bytes(merged.SerializeToString())
    new_tok = export_hf(path, tmp / "merged_emb_hf")
    cfg = Gemma3TextConfig(vocab_size=len(base_tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                           layer_types=["full_attention"])
    model = Gemma3ForCausalLM(cfg)
    before = model.get_input_embeddings().weight.detach().clone()
    stats = resize_and_init(model, base_tok, new_tok, manifest["new_token_ids"], pad_to_multiple_of=8)
    after = model.get_input_embeddings().weight
    assert after.shape[0] >= len(new_tok) and after.shape[0] % 8 == 0
    assert torch.equal(after[: before.shape[0]], before)          # old rows preserved
    tid = manifest["new_token_ids"][0]
    sub = base_tok(new_tok.convert_tokens_to_string([new_tok.convert_ids_to_tokens(tid)]),
                   add_special_tokens=False)["input_ids"]
    assert torch.allclose(after[tid], before[sub].mean(0), atol=1e-6)
    assert model.get_output_embeddings().weight.data_ptr() == after.data_ptr()  # still tied
    assert stats["initialized"] == len(manifest["new_token_ids"])


def test_sft_masks_everything_but_assistant(tokenizers):
    _, _, _, tok = tokenizers
    msgs = row_to_messages({"instruction": "the cat", "input": "", "output": "sat on the mat"}, "alpaca", {})
    ex = encode_conversation(tok, msgs, max_len=512)
    trained = [t for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != IGNORE]
    assert tok.decode(trained) == "sat on the mat<end_of_turn>"
    full = tok.decode(ex["input_ids"])
    assert full == tok.apply_chat_template(msgs, tokenize=False)
    batch = ChatCollator(tok.pad_token_id)([ex, {"input_ids": ex["input_ids"][:5], "labels": ex["labels"][:5]}])
    assert batch["input_ids"].shape[1] % 8 == 0 and batch["attention_mask"][1].sum() == 5
    assert encode_conversation(tok, msgs, max_len=4) is None


def test_mixture_plan_and_selection():
    import numpy as np

    plan = compute_plan({"ta": 1000, "en": 5000, "tiny": 10}, {"ta": 0.7, "en": 0.2, "tiny": 0.1, "none": 0.1},
                        None, {"en": 1.0}, 4.0)
    assert "none" not in plan
    assert plan["ta"]["epochs"] == pytest.approx(1.0)
    assert plan["tiny"]["target_tokens"] == 40                       # capped at 4 epochs
    idx = _select(np.array([10, 20, 30, 40]), 250, np.random.default_rng(0))
    assert np.array([10, 20, 30, 40])[idx].sum() >= 250 and len(idx) >= 8


def test_label_only_loss_matches_standard_loss():
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    from training.common import label_only_loss
    from training.model_factory import apply_trainable

    torch.manual_seed(0)
    cfg = Gemma3TextConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                           layer_types=["sliding_attention", "full_attention"], final_logit_softcapping=30.0)
    model = Gemma3ForCausalLM(cfg).eval()
    ids = torch.randint(4, 97, (3, 12))
    labels = ids.clone()
    labels[:, :5] = -100
    labels[1, 8:] = -100
    mask = torch.ones_like(ids)
    ref = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    ours = label_only_loss(model, ids, mask, labels)
    assert torch.allclose(ref, ours, atol=1e-5)

    info = apply_trainable(model, "no_embeddings")
    assert not model.get_input_embeddings().weight.requires_grad
    assert info["trainable_params"] == sum(p.numel() for n, p in model.named_parameters() if "embed" not in n)
