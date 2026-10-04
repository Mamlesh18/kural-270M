"""Human-eval summary maths and the evaluation server's HTTP API (with a stub model pool)."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from evaluation.human_eval import append_rating, load_ratings, summarize
from inference.server import Handler


def test_summary_win_rates_and_scores(tmp_path):
    f = tmp_path / "r.jsonl"
    append_rating(f, {"mode": "compare", "category": "factual", "model_a": "kural", "model_b": "gemma",
                      "winner": "a", "scores": {"a": {"correctness": 5}, "b": {"correctness": 1}}})
    append_rating(f, {"mode": "compare", "category": "factual", "model_a": "gemma", "model_b": "kural", "winner": "tie"})
    append_rating(f, {"mode": "compare", "category": "tanglish", "model_a": "gemma", "model_b": "kural", "winner": "both_bad"})
    append_rating(f, {"mode": "chat", "category": "custom", "model": "kural", "scores": {"fluency": 4}})
    s = summarize(load_ratings(f))
    k, g = s["models"]["kural"], s["models"]["gemma"]
    assert (k["comparisons"], k["wins"], k["ties"], k["both_bad"], k["losses"]) == (3, 1, 1, 1, 0)
    assert k["win_rate"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert g["win_rate"] == pytest.approx(1 / 3)
    assert k["scores"] == {"correctness": 5.0, "fluency": 4.0}
    assert s["ranking"][0] == "kural"
    assert s["by_category"]["factual"]["kural"]["comparisons"] == 2
    assert s["head_to_head"] == [{"model_x": "gemma", "model_y": "kural", "x_wins": 0, "y_wins": 1, "ties": 2}]


@pytest.mark.parametrize("bad", [
    {"mode": "compare", "model_a": "a", "model_b": "b", "winner": "maybe"},
    {"mode": "compare", "model_a": "a", "winner": "a"},
    {"mode": "chat", "model": "m", "scores": {"fluency": 9}},
    {"mode": "chat", "model": "m", "scores": {"style": 3}},
    {"mode": "other"},
])
def test_invalid_ratings_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        append_rating(tmp_path / "r.jsonl", bad)


class StubPool:
    device = "cpu"
    specs = {"echo": {"path": "x", "chat": True}}

    def describe(self):
        return [{"name": "echo", "path": "x", "chat": True, "description": "", "available": True, "loaded": True}]

    def stream(self, name, messages, params, stop):
        for word in messages[-1]["content"].split():
            yield {"text": word + " "}
        yield {"done": True, "stats": {"new_tokens": 2, "seconds": 0.0, "tokens_per_second": None, "params": params}}


@pytest.fixture
def server(tmp_path):
    handler = type("H", (Handler,), {"pool": StubPool(), "ratings_file": tmp_path / "ratings.jsonl",
                                     "prompts": [{"id": "p1", "category": "factual", "prompt": "q"}],
                                     "defaults": {"max_new_tokens": 8}, "title": "t"})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=10)


def test_server_endpoints(server):
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "Blind compare" in html and "Run test set" in html
    cfg = json.load(urllib.request.urlopen(server + "/api/config"))
    assert cfg["models"][0]["name"] == "echo" and "correctness" in cfg["criteria"]
    assert json.load(urllib.request.urlopen(server + "/api/prompts"))[0]["id"] == "p1"

    lines = _post(server + "/api/generate", {"model": "echo", "messages": [{"role": "user", "content": "வணக்கம் நண்பா"}],
                                             "params": {"temperature": 0}}).read().decode("utf-8").splitlines()
    events = [json.loads(x) for x in lines if x.strip()]
    assert "".join(e.get("text", "") for e in events) == "வணக்கம் நண்பா "
    assert events[-1]["done"] and events[-1]["stats"]["params"]["max_new_tokens"] == 8   # defaults merged

    with pytest.raises(urllib.error.HTTPError) as err:
        _post(server + "/api/generate", {"model": "nope", "messages": [{"role": "user", "content": "x"}]})
    assert err.value.code == 400

    assert _post(server + "/api/rate", {"mode": "compare", "model_a": "echo", "model_b": "other", "winner": "a",
                                        "category": "factual"}).status == 200
    summary = json.load(urllib.request.urlopen(server + "/api/summary"))
    assert summary["n_compare"] == 1 and summary["models"]["echo"]["wins"] == 1
    raw = urllib.request.urlopen(server + "/api/ratings.jsonl").read().decode("utf-8")
    assert json.loads(raw)["winner"] == "a"
