"""Serve a fixed PTX candidate through the OpenAI Responses API surface."""

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import time
from urllib.parse import urlparse
from uuid import uuid4


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a local OpenAI-compatible endpoint that returns fixed PTX."
    )
    parser.add_argument(
        "--ptx",
        type=Path,
        required=True,
        help="PTX file returned for initial-candidate requests.",
    )
    parser.add_argument("--num-threads-x", type=int, default=256)
    parser.add_argument("--num-threads-y", type=int, default=1)
    parser.add_argument("--num-threads-z", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def _response_body(request_body, candidate):
    response_format = request_body.get("text", {}).get("format", {})
    format_name = response_format.get("name")
    if (
        format_name == "ptx_kernel_metadata"
        and "previous_response_id" not in request_body
    ):
        return _patch_response(request_body, candidate)
    if format_name == "ptx_kernel":
        output_text = json.dumps(
            {
                **candidate,
                "difficulties": [],
            }
        )
    elif format_name == "failure_analysis":
        output_text = json.dumps(
            {
                "root_cause": "abc",
                "repair_instruction": "abc",
            }
        )
    elif format_name == "improvement_plan":
        output_text = json.dumps(
            {
                "improvements": [
                    {
                        "name": "abc",
                        "rationale": "abc",
                        "instruction": "abc",
                    }
                ]
            }
        )
    else:
        output_text = json.dumps(
            {
                **{key: value for key, value in candidate.items() if key != "ptx"},
                "difficulties": [],
            }
        )

    return {
        "id": f"resp_fake_{uuid4().hex}",
        "object": "response",
        "created_at": int(time()),
        "model": request_body.get("model", "fake-ptx"),
        "output": [
            {
                "id": f"msg_fake_{uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": output_text,
                    }
                ],
            }
        ],
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "parallel_tool_calls": True,
        "store": False,
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }


def _patch_response(request_body, candidate):
    patch = (
        "--- candidate.ptx\n"
        "+++ candidate.ptx\n"
        "@@ -13,1 +13,1 @@\n"
        "-    // Allocate two shared stages so copies overlap tensor-core work.\n"
        "+    // abc"
    )
    return {
        "id": f"resp_fake_{uuid4().hex}",
        "object": "response",
        "created_at": int(time()),
        "model": request_body.get("model", "fake-ptx"),
        "output": [
            {
                "id": f"fc_fake_{uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "name": "apply_ptx_patch",
                "call_id": f"call_fake_{uuid4().hex}",
                "arguments": json.dumps(
                    {
                        "patch": patch,
                        **{
                            key: value
                            for key, value in candidate.items()
                            if key != "ptx" and key != "difficulties"
                        },
                    }
                ),
            }
        ],
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "parallel_tool_calls": True,
        "store": False,
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }


def _handler(candidate):
    class FakeOpenAIHandler(BaseHTTPRequestHandler):
        def _send_json(self, status, payload):
            serialized_payload = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(serialized_payload)))
            self.end_headers()
            self.wfile.write(serialized_payload)

        def do_GET(self):
            if urlparse(self.path).path == "/v1/skills":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [],
                        "has_more": False,
                        "first_id": None,
                        "last_id": None,
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length)
            request_path = urlparse(self.path).path
            if request_path == "/v1/skills":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "id": f"skill_fake_{uuid4().hex}",
                        "object": "skill",
                        "name": "fake-skill",
                    },
                )
                return
            if request_path != "/v1/responses":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
            try:
                parsed_request = json.loads(request_body)
            except json.JSONDecodeError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON."})
                return
            self._send_json(HTTPStatus.OK, _response_body(parsed_request, candidate))

        def log_message(self, format_string, *args):
            del format_string, args

    return FakeOpenAIHandler


def main():
    args = _parse_args()
    candidate = {
        "ptx": args.ptx.read_text(encoding="utf-8"),
        "num_threads_x": args.num_threads_x,
        "num_threads_y": args.num_threads_y,
        "num_threads_z": args.num_threads_z,
        "difficulties": [],
    }
    server = ThreadingHTTPServer((args.host, args.port), _handler(candidate))
    print(f"Serving fake OpenAI endpoint at http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
