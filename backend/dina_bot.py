from __future__ import annotations

import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

DEFAULT_PORT = int(os.environ.get("PORT", "3000"))
KB_DIR = Path(os.environ.get("DINA_KB_DIR", str(Path(__file__).parent.parent / "kb"))).resolve()
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODEL_NAME = os.environ.get("DINA_MODEL", "gpt-4.1-mini")


SYSTEM_PROMPT = (
    "You are Dina Bot, a minimal assistant for answering questions about Dina. "
    "Use only the provided context from markdown files. "
    "If the answer is not in the context, say you do not know. "
    "Keep responses short and clear."
)


@dataclass
class DocChunk:
    source: str
    text: str


def _read_markdown_files() -> list[DocChunk]:
    if not KB_DIR.exists():
        return []
    chunks: list[DocChunk] = []
    for path in KB_DIR.rglob("*.md"):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        chunks.append(DocChunk(source=str(path.relative_to(KB_DIR)), text=content))
    return chunks


def _score_chunk(chunk: DocChunk, query_terms: set[str]) -> int:
    lowered = chunk.text.lower()
    return sum(1 for term in query_terms if term in lowered)


def _pick_relevant(chunks: Iterable[DocChunk], query: str, limit: int = 3) -> list[DocChunk]:
    terms = {term for term in query.lower().split() if len(term) > 2}
    if not terms:
        return []
    scored = sorted(chunks, key=lambda c: _score_chunk(c, terms), reverse=True)
    return [chunk for chunk in scored if _score_chunk(chunk, terms) > 0][:limit]


def _build_context(chunks: list[DocChunk]) -> str:
    if not chunks:
        return "No relevant context found."
    blocks = []
    for chunk in chunks:
        blocks.append(f"Source: {chunk.source}\n{chunk.text.strip()}")
    return "\n\n---\n\n".join(blocks)


class DinaBotHandler(BaseHTTPRequestHandler):
    server_version = "DinaBotHTTP/0.1"

    def _send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self) -> None:
        self._send_json(204, {})

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self._send_json(404, {"error": "Not found"})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/" or path == "/index.html":
            self._serve_file(FRONTEND_DIR / "index.html", "text/html")
            return
        if path == "/styles.css":
            self._serve_file(FRONTEND_DIR / "styles.css", "text/css")
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/chat":
            self._send_json(404, {"error": "Not found"})
            return

        payload = self._read_json()
        user_message = payload.get("message", "").strip()

        if not user_message:
            self._send_json(
                400,
                {"error": "Message is required", "example": {"message": "Hi Dina Bot"}},
            )
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self._send_json(
                500,
                {"error": "Missing OPENAI_API_KEY in environment"},
            )
            return

        chunks = _read_markdown_files()
        relevant = _pick_relevant(chunks, user_message)
        context = _build_context(relevant)

        client = OpenAI(api_key=api_key)
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Question: {user_message}\n\nContext:\n{context}",
                    },
                ],
            )
        except Exception as exc:
            self._send_json(
                503,
                {"error": "Sorry, I'm having trouble connecting right now. Please try again later.", "detail": str(exc)},
            )
            return

        self._send_json(
            200,
            {
                "reply": response.output_text,
                "received": user_message,
                "sources": [chunk.source for chunk in relevant],
            },
        )


def run_server(port: int = DEFAULT_PORT) -> None:
    server = HTTPServer(("", port), DinaBotHandler)
    print(f"Dina Bot backend running at http://localhost:{port}")
    print(f"KB directory: {KB_DIR}")
    print("Endpoints: GET /health, POST /chat")
    server.serve_forever()


if __name__ == "__main__":
    run_server()