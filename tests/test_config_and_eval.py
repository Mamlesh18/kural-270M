from pathlib import Path

import pytest
import torch

from common.config import REPO_ROOT, load_config
from evaluation.perplexity import document_nll

CONFIGS = sorted(p for p in (REPO_ROOT / "configs").rglob("*.yaml")
                 if p.name not in ("common.yaml", "benchmarks.yaml") and "model" not in p.parts
                 and not p.name.startswith("mixture_"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_config_resolves(path, monkeypatch, tmp_path):
    monkeypatch.setenv("KURAL_WORKSPACE", str(tmp_path))
    cfg = load_config(path)
    assert cfg.paths.workspace == str(tmp_path)


def test_overrides_and_inheritance(tmp_path, monkeypatch):
    monkeypatch.setenv("KURAL_WORKSPACE", str(tmp_path))
    cfg = load_config(REPO_ROOT / "configs/train/cpt_270m_stage2_full.yaml", ["train.args.learning_rate=1e-5"])
    assert cfg.model.source == "checkpoint"
    assert cfg.train.args.learning_rate == 1e-5
    assert cfg.train.args.gradient_accumulation_steps == 4   # inherited from cpt_270m.yaml
    assert Path(cfg.data.packed_dir).parent == Path(tmp_path) / "packed"


class _Uniform(torch.nn.Module):
    """Predicts a uniform distribution → NLL per token is ln(V) regardless of window."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        return type("O", (), {"logits": torch.zeros(ids.shape[0], ids.shape[1], self.vocab)})()


@pytest.mark.parametrize("max_len,stride", [(8, 4), (8, 8), (5, 1), (64, 32)])
def test_sliding_window_scores_every_token_once(max_len, stride):
    ids = list(range(1, 30))
    nll, count = document_nll(_Uniform(50), ids, max_len, stride, "cpu")
    assert count == len(ids) - 1
    assert nll == pytest.approx(count * torch.log(torch.tensor(50.0)).item(), rel=1e-5)
