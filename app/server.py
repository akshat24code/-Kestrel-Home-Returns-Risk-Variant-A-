"""Tiny HTTP service, standard library only.

  GET  /           the screen (static/index.html)
  POST /score      one order as JSON -> probability, action, reasons
  GET  /health     liveness + model info
  GET  /examples   a few real dispatch-time records to try

Run: python -m app.server   (PORT env var, default 8000)
No API keys, no network calls, no per-order cost.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.scorer import Scorer  # noqa: E402

MODEL_PATH = ROOT / "model" / "model.joblib"
if not MODEL_PATH.exists():
    sys.exit("model/model.joblib not found. Run `python -m src.train` first (see README).")
SCORER = Scorer(MODEL_PATH)
INDEX = (ROOT / "app" / "static" / "index.html").read_bytes()


def load_examples():
    return json.loads((ROOT / "app" / "static" / "examples.json").read_text())


EXAMPLES = load_examples()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, INDEX, "text/html; charset=utf-8")
        if self.path == "/health":
            return self._send(200, {"status": "ok", **SCORER.meta})
        if self.path == "/examples":
            return self._send(200, EXAMPLES)
        self._send(404, {"ok": False, "errors": ["not found"]})

    def do_POST(self):
        if self.path != "/score":
            return self._send(404, {"ok": False, "errors": ["not found"]})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 100_000:
                return self._send(413, {"ok": False, "errors": ["record too large"]})
            rec = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(rec, dict):
                return self._send(400, {"ok": False, "errors": ["send one order as a JSON object"]})
        except json.JSONDecodeError:
            return self._send(400, {"ok": False, "errors": ["body is not valid JSON"]})
        try:
            out = SCORER.score(rec)
        except Exception as e:  # never leak a stack trace to the screen
            return self._send(500, {"ok": False, "errors": [f"could not score this order: {type(e).__name__}"]})
        self._send(200 if out["ok"] else 422, out)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.command, fmt % args))


def main():
    port = int(os.environ.get("PORT", 8000))
    print(f"Kestrel returns-risk service on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
