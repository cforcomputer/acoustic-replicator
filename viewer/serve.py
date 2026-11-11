"""Local viewer server.  Serves the project folder on http://127.0.0.1:8791 with
caching disabled so the page always shows the simulation's latest output.

    .venv/bin/python viewer/serve.py            then open  http://127.0.0.1:8791/viewer/
"""
import http.server, json, os, sys
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8791


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/api/list"):
            # list the physics-replay files so the page can discover new ones
            d = os.path.join(ROOT, "out", "viewer")
            names = sorted(f for f in os.listdir(d) if f.startswith("replay_") and f.endswith(".json")) if os.path.isdir(d) else []
            body = json.dumps(names).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", ""):
            self.send_response(302); self.send_header("Location", "/viewer/"); self.end_headers(); return
        super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=ROOT))
    print(f"viewer at http://127.0.0.1:{PORT}/viewer/", flush=True)
    srv.serve_forever()
