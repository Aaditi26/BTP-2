import json
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from optimization_engine import optimize_job


class OptimizationRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/optimize":
            self._send_json({"error": "Route not found."}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            job_data = json.loads(raw_body.decode("utf-8"))
            result = optimize_job(job_data)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON payload."}, status=HTTPStatus.BAD_REQUEST)
            return
        except KeyError as exc:
            self._send_json({"error": f"Missing required field: {exc.args[0]}"}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._send_json({"error": f"Optimization failed: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(result)


def main() -> None:
    root = Path(__file__).resolve().parent
    host = "127.0.0.1"
    port = 8000

    print(f"Serving {root} at http://{host}:{port}")
    handler = lambda *args, **kwargs: OptimizationRequestHandler(*args, directory=str(root), **kwargs)
    with ThreadingHTTPServer((host, port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
