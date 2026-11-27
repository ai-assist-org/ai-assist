"""Local HTTPS origin for the sandbox proxy integration test."""

import json
import os
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

release = threading.Event()


class Origin(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.do_POST()

    def do_POST(self) -> None:
        path, _, query = self.path.partition("?")
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path.startswith("/repo.git/"):
            env = dict(
                os.environ,
                GIT_PROJECT_ROOT="/tmp",
                GIT_HTTP_EXPORT_ALL="1",
                PATH_INFO=path,
                REQUEST_METHOD=self.command,
                QUERY_STRING=query,
                CONTENT_TYPE=self.headers.get("Content-Type", ""),
                CONTENT_LENGTH=str(len(body)),
                REMOTE_USER="fixture",
            )
            out = subprocess.run(["git", "http-backend"], input=body, env=env, capture_output=True, check=True).stdout
            header, body = out.split(b"\r\n\r\n", 1)
            self.send_response(200)
            for line in header.decode().splitlines():
                key, value = line.split(": ", 1)
                self.send_header(key, value)
        elif path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = json.dumps(
                {
                    "id": "fixture",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini",
                    "created": 1,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "fixture reply"},
                        }
                    ],
                }
            ).encode()
        else:
            self.send_response(200)
        if path == "/events":
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: first\n\n")
            self.wfile.flush()
            if release.wait(8):
                self.wfile.write(b"data: last\n\n")
            return
        if path == "/release":
            release.set()
        self.end_headers()
        self.wfile.write(body)


subprocess.run(["git", "init", "--bare", "/tmp/repo.git"], check=True)
cert = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
cert.load_cert_chain("/fixture/server.pem", "/fixture/server.key")


class TLSOrigin(ThreadingHTTPServer):
    def get_request(self):
        conn, peer = super().get_request()
        with Path("/tmp/accepted").open("ab") as log:
            log.write(b".")
        return cert.wrap_socket(conn, server_side=True), peer


server = TLSOrigin(("", 443), Origin)
server.serve_forever()
