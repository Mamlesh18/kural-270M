"""Text generation / chat with a trained (optionally quantized) Kural model.

    python -m inference.generate --config configs/inference/generate.yaml inference.prompt="தமிழ்நாட்டின்"
    python -m inference.generate --config configs/inference/generate.yaml inference.chat=true inference.interactive=true

``inference.model_path`` may point to an HF model directory or to a
``dynamic_int8`` export from ``inference.quantize`` (``quantized_model.pt``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from common.config import parse_config, to_container
from common.utils import get_logger, set_seed, setup_logging, torch_dtype

log = get_logger("inference.generate")


def load_for_inference(path: str, dtype: str | None = None, device: str | None = None,
                       load_in_4bit: bool = False, load_in_8bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = Path(path)
    tok = AutoTokenizer.from_pretrained(path)
    if (p / "quantized_model.pt").exists():
        # torch dynamic-int8 export (CPU only); pickled full module.
        model = torch.load(p / "quantized_model.pt", weights_only=False)
        return model.eval(), tok, "cpu"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kw = {}
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_quant_type="nf4")
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype(dtype), **kw)
    if not kw:
        model = model.to(device)
    return model.eval(), tok, device


def build_inputs(tok, prompt: str, chat: bool, history: list[dict[str, str]] | None = None,
                 system: str | None = None):
    if chat:
        msgs = ([{"role": "system", "content": system}] if system else []) + (history or [])
        msgs = msgs + [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", add_special_tokens=False)
    return tok(prompt, return_tensors="pt")


@torch.no_grad()
def generate(model, tok, device, prompt: str, chat: bool = False, stream: bool = True,
             history=None, system=None, **gen) -> str:
    from transformers import TextStreamer

    enc = {k: v.to(device) for k, v in build_inputs(tok, prompt, chat, history, system).items()}
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None
    out = model.generate(**enc, streamer=streamer, pad_token_id=tok.pad_token_id, **gen)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run(cfg) -> None:
    c = to_container(cfg.inference)
    set_seed(int(cfg.seed))
    model, tok, device = load_for_inference(c["model_path"], c.get("dtype"), c.get("device"),
                                            bool(c.get("load_in_4bit", False)), bool(c.get("load_in_8bit", False)))
    gen = dict(c.get("generation") or {})
    chat = bool(c.get("chat", False))
    if c.get("interactive"):
        history: list[dict[str, str]] = []
        print("Kural — type /exit to quit, /reset to clear history")
        while True:
            try:
                prompt = input("\n>>> ").strip()
            except EOFError:
                break
            if prompt in ("/exit", "/quit"):
                break
            if prompt == "/reset":
                history = []
                continue
            reply = generate(model, tok, device, prompt, chat, True, history, c.get("system_prompt"), **gen)
            if chat:
                history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply.strip()}]
        return
    prompts = c.get("prompts") or [c["prompt"]]
    for p in prompts:
        print(f"\n### {p}")
        generate(model, tok, device, p, chat, True, None, c.get("system_prompt"), **gen)


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Generate text with a Kural model", argv))


if __name__ == "__main__":
    main()
