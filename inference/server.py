"""Local chatbot evaluation server: serves ``inference/web/eval_chat.html`` and a small JSON API.

    python -m inference.server --config configs/inference/eval_server.yaml
    # then open http://127.0.0.1:7860

Models listed in ``server.models`` are loaded lazily on first use and kept in
memory. Each model has its own lock, so two models can generate at once (blind
A/B comparison) while requests to the same model queue up.

API (all JSON):

    GET  /api/config        models, default generation params, rating criteria
    GET  /api/prompts       the evaluation prompt set
    POST /api/generate      {model, messages, params} → NDJSON stream of {"text"} … {"done", stats}
    POST /api/rate          one rating record (see evaluation/human_eval.py) → appended to ratings_file
    GET  /api/summary       aggregated win rates / scores
    GET  /api/ratings.jsonl raw ratings download

Only the standard library is used for HTTP. Binds to 127.0.0.1 by default —
there is no authentication, so do not expose it publicly.
"""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import torch

from common.config import parse_config, to_container
from common.utils import get_logger, read_jsonl, setup_logging, torch_dtype
from evaluation.human_eval import CRITERIA, append_rating, load_ratings, summarize
from tokenizer.train_tokenizer import DEFAULT_CHAT_TEMPLATE

log = get_logger("inference.server")
WEB_DIR = Path(__file__).resolve().parent / "web"
GEN_KEYS = {"max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "do_sample"}


def _is_available(path: str) -> bool:
    p = Path(path)
    if p.exists():
        return True
    # Hub ids look like "org/name" and are not existing local paths.
    return "/" in path and not p.is_absolute() and not path.startswith((".", "~")) and path.count("/") == 1


class _Stop:
    """StoppingCriteria that fires when the client disconnects."""

    def __init__(self):
        self.event = threading.Event()

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class ModelPool:
    def __init__(self, specs: list[dict[str, Any]], device: str | None, dtype: str | None, max_context: int):
        self.specs = {s["name"]: s for s in specs}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_context = max_context
        self._models: dict[str, tuple[Any, Any]] = {}
        self._locks = {name: threading.Lock() for name in self.specs}
        self._load_lock = threading.Lock()

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": n, "path": s["path"], "chat": bool(s.get("chat", True)),
                 "description": s.get("description", ""), "available": _is_available(s["path"]),
                 "loaded": n in self._models} for n, s in self.specs.items()]

    def get(self, name: str):
        if name not in self.specs:
            raise KeyError(f"unknown model {name!r}")
        with self._load_lock:
            if name not in self._models:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                path = self.specs[name]["path"]
                t0 = time.time()
                log.info("Loading %s from %s ...", name, path)
                tok = AutoTokenizer.from_pretrained(path)
                if (Path(path) / "quantized_model.pt").exists():
                    # dynamic-int8 export from inference.quantize (CPU only, pickled module)
                    model = torch.load(Path(path) / "quantized_model.pt", weights_only=False).eval()
                else:
                    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype(self.dtype))
                    model.to(self.device).eval()
                if tok.chat_template is None:
                    tok.chat_template = DEFAULT_CHAT_TEMPLATE
                self._models[name] = (model, tok)
                log.info("Loaded %s in %.1fs", name, time.time() - t0)
        return self._models[name]

    def _prompt_ids(self, name: str, tok, messages: list[dict[str, str]]) -> torch.Tensor:
        if self.specs[name].get("chat", True):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            # Base (non-instruction) model: plain continuation of the conversation text.
            text = "\n\n".join(m["content"] for m in messages if m["role"] != "system") + "\n"
            ids = tok(text, return_tensors="pt")["input_ids"]
        return ids[:, -self.max_context:]

    def stream(self, name: str, messages: list[dict[str, str]], params: dict[str, Any],
               stop: _Stop) -> Iterator[dict[str, Any]]:
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        model, tok = self.get(name)
        ids = self._prompt_ids(name, tok, messages).to(next(model.parameters()).device)
        gen = {k: v for k, v in params.items() if k in GEN_KEYS and v is not None}
        gen.setdefault("max_new_tokens", 256)
        gen["max_new_tokens"] = int(min(int(gen["max_new_tokens"]), 2048))
        if not gen.get("do_sample", True) or float(gen.get("temperature", 1.0) or 0) <= 0:
            gen["do_sample"] = False
            for k in ("temperature", "top_p", "top_k"):
                gen.pop(k, None)
        eos = [tok.eos_token_id]
        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot != tok.unk_token_id:
            eos.append(eot)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=600)
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), streamer=streamer, eos_token_id=eos,
                      pad_token_id=tok.pad_token_id, stopping_criteria=StoppingCriteriaList([stop]), **gen)
        errors: list[BaseException] = []

        def work():
            try:
                with torch.no_grad():
                    out = model.generate(**kwargs)
                kwargs["_n_new"] = int(out.shape[1] - ids.shape[1])
            except BaseException as exc:  # surfaced to the client below
                errors.append(exc)
                streamer.end()

        with self._locks[name]:
            t0 = time.time()
            th = threading.Thread(target=work, daemon=True)
            th.start()
            for piece in streamer:
                if piece:
                    yield {"text": piece}
            th.join()
            if errors:
                raise errors[0]
            dt = time.time() - t0
            n = kwargs.get("_n_new", 0)
            yield {"done": True, "stats": {"new_tokens": n, "seconds": round(dt, 2),
                                           "tokens_per_second": round(n / dt, 1) if dt > 0 else None,
                                           "prompt_tokens": int(ids.shape[1]), "params": gen}}


class Handler(BaseHTTPRequestHandler):
    server_version = "KuralEval/1.0"
    pool: ModelPool
    ratings_file: Path
    prompts: list[dict[str, Any]]
    defaults: dict[str, Any]
    title: str

    def log_message(self, fmt, *args):  # route through logging, quieter
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 5_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                body = (WEB_DIR / "eval_chat.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/config":
                self._json({"title": self.title, "models": self.pool.describe(), "defaults": self.defaults,
                            "criteria": CRITERIA, "device": self.pool.device})
            elif path == "/api/prompts":
                self._json(self.prompts)
            elif path == "/api/summary":
                self._json(summarize(load_ratings(self.ratings_file)))
            elif path == "/api/ratings.jsonl":
                body = self.ratings_file.read_bytes() if self.ratings_file.exists() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="ratings.jsonl"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            log.exception("GET %s failed", path)
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            data = self._body()
        except Exception as exc:
            return self._json({"error": f"bad request: {exc}"}, HTTPStatus.BAD_REQUEST)
        if path == "/api/rate":
            try:
                rec = append_rating(self.ratings_file, data)
                return self._json({"ok": True, "ts": rec["ts"]})
            except ValueError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if path == "/api/generate":
            return self._generate(data)
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _generate(self, data: dict[str, Any]) -> None:
        name = data.get("model")
        messages = data.get("messages") or []
        if name not in self.pool.specs:
            return self._json({"error": f"unknown model {name!r}"}, HTTPStatus.BAD_REQUEST)
        if not messages or messages[-1].get("role") != "user":
            return self._json({"error": "messages must end with a user turn"}, HTTPStatus.BAD_REQUEST)
        messages = [{"role": m["role"], "content": str(m.get("content", ""))} for m in messages
                    if m.get("role") in ("system", "user", "assistant")]
        params = {**self.defaults, **(data.get("params") or {})}
        stop = _Stop()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj: dict[str, Any]) -> None:
            chunk = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()

        try:
            for event in self.pool.stream(name, messages, params, stop):
                send(event)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            stop.event.set()
            log.info("Client disconnected; generation for %s stopped", name)
            return
        except Exception as exc:
            log.exception("generation failed")
            try:
                send({"error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                return
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass


def build_server(cfg) -> ThreadingHTTPServer:
    s = to_container(cfg.server)
    specs = [m for m in s["models"] if m.get("enabled", True)]
    pool = ModelPool(specs, s.get("device"), s.get("dtype"), int(s.get("max_context", 2048)))
    prompts = list(read_jsonl(s["prompts_file"])) if s.get("prompts_file") and Path(s["prompts_file"]).exists() else []
    attrs = {"pool": pool, "ratings_file": Path(s["ratings_file"]), "prompts": prompts,
             "defaults": dict(s.get("generation") or {}), "title": s.get("title", "Kural evaluation")}
    handler = type("KuralHandler", (Handler,), attrs)
    httpd = ThreadingHTTPServer((s.get("host", "127.0.0.1"), int(s.get("port", 7860))), handler)
    httpd.daemon_threads = True
    for m in pool.describe():
        log.info("  model %-28s %-9s %s", m["name"], "ok" if m["available"] else "MISSING", m["path"])
    for name in s.get("preload") or []:
        pool.get(name)
    return httpd


def main(argv=None) -> None:
    setup_logging()
    cfg = parse_config("Local chatbot evaluation server", argv)
    httpd = build_server(cfg)
    host, port = httpd.server_address[:2]
    log.info("Open http://%s:%d  (ratings → %s)", host, port, cfg.server.ratings_file)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
