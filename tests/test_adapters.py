from __future__ import annotations

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from divy.adapters import HttpModelAdapter, ProviderError
from divy.performance import PerformanceMemory, PerformanceObservation


class FakeProviderHandler(BaseHTTPRequestHandler):
    mode = "success"

    def do_POST(self) -> None:  # noqa: N802 - standard library handler API
        length = int(self.headers.get("Content-Length", "0"))
        request_body = json.loads(self.rfile.read(length))
        self.server.last_request = request_body  # type: ignore[attr-defined]
        if self.mode == "timeout":
            time.sleep(0.15)
        if self.mode == "rate":
            self.send_response(429)
            self.end_headers()
            return
        if self.mode == "auth":
            self.send_response(401)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if self.mode == "malformed":
            self.wfile.write(b"not-json")
            return
        payload = {
            "choices": [{"message": {"content": json.dumps({"answer": "ok"})}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }
        try:
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        except (BrokenPipeError, ConnectionAbortedError):
            return

    def log_message(self, format: str, *args) -> None:
        return


class AdapterTests(unittest.TestCase):
    def _server(self, mode: str):
        FakeProviderHandler.mode = mode
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProviderHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_success_parses_structured_content_and_usage(self) -> None:
        server, thread = self._server("success")
        try:
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model", "secret-not-logged")
            response = adapter.generate([{"role": "user", "content": "hello"}], [{"name": "fs.read"}], {"type": "object"})
            self.assertEqual(response.content, {"answer": "ok"})
            self.assertEqual(response.token_usage, {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7})
            self.assertEqual(server.last_request["model"], "test-model")  # type: ignore[attr-defined]
            self.assertFalse(server.last_request["stream"])  # type: ignore[attr-defined]
            self.assertIn("tools", server.last_request)  # type: ignore[attr-defined]
            self.assertEqual(server.last_request["tools"][0]["type"], "function")  # type: ignore[attr-defined]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_function_tool_call_response_is_normalized_for_divy(self) -> None:
        original_mode = FakeProviderHandler.mode
        original_do_post = FakeProviderHandler.do_POST
        server, thread = self._server("success")
        try:
            def tool_call(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "fs.read", "arguments": '{"path":"x.txt"}'}}]}}]}
                try:
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                except (BrokenPipeError, ConnectionAbortedError):
                    return

            FakeProviderHandler.do_POST = tool_call
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model")
            response = adapter.generate([], [{"name": "fs.read", "description": "read", "parameters": {"type": "object"}}], {})
            self.assertEqual(response.content["action"], {"tool": "fs.read", "arguments": {"path": "x.txt"}})
        finally:
            FakeProviderHandler.do_POST = original_do_post
            FakeProviderHandler.mode = original_mode
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_pseudo_final_tool_call_is_normalized_as_final_candidate(self) -> None:
        original_mode = FakeProviderHandler.mode
        original_do_post = FakeProviderHandler.do_POST
        server, thread = self._server("success")
        try:
            def pseudo_final(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1", "function": {"name": "final", "arguments": '{"message":"done"}'}}]}}]}
                self.wfile.write(json.dumps(payload).encode("utf-8"))

            FakeProviderHandler.do_POST = pseudo_final
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model")
            response = adapter.generate([], [{"name": "final", "description": "reserved", "parameters": {"type": "object"}}], {})
            self.assertEqual(response.content, {"final": {"message": "done"}})
        finally:
            FakeProviderHandler.do_POST = original_do_post
            FakeProviderHandler.mode = original_mode
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_provider_safe_tool_name_round_trips_to_divy_name(self) -> None:
        original_mode = FakeProviderHandler.mode
        original_do_post = FakeProviderHandler.do_POST
        server, thread = self._server("success")
        try:
            def tool_call(self):
                length = int(self.headers.get("Content-Length", "0"))
                request_body = json.loads(self.rfile.read(length))
                self.server.last_request = request_body  # type: ignore[attr-defined]
                provider_name = request_body["tools"][0]["function"]["name"]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1", "function": {"name": provider_name, "arguments": "{}"}}]}}]}
                self.wfile.write(json.dumps(payload).encode("utf-8"))

            FakeProviderHandler.do_POST = tool_call
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model")
            response = adapter.generate([], [{"name": "fs.read", "description": "read", "parameters": {"type": "object"}}], {})
            self.assertNotEqual(server.last_request["tools"][0]["function"]["name"], "fs.read")  # type: ignore[attr-defined]
            self.assertEqual(response.content["action"]["tool"], "fs.read")
        finally:
            FakeProviderHandler.do_POST = original_do_post
            FakeProviderHandler.mode = original_mode
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_detokenized_provider_tool_alias_round_trips_to_divy_name(self) -> None:
        original_mode = FakeProviderHandler.mode
        original_do_post = FakeProviderHandler.do_POST
        server, thread = self._server("success")
        try:
            def tool_call(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1", "function": {"name": "fs.x2e_write", "arguments": '{"path":"x"}'}}]}}]}
                self.wfile.write(json.dumps(payload).encode("utf-8"))

            FakeProviderHandler.do_POST = tool_call
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model")
            response = adapter.generate([], [{"name": "fs.write", "description": "write", "parameters": {"type": "object"}}], {})
            self.assertEqual(response.content["action"]["tool"], "fs.write")
        finally:
            FakeProviderHandler.do_POST = original_do_post
            FakeProviderHandler.mode = original_mode
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_internal_tool_conversation_is_normalized_for_provider(self) -> None:
        server, thread = self._server("success")
        try:
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model")
            adapter.generate(
                [
                    {"role": "user", "content": "read"},
                    {"role": "assistant", "content": {"action": {"tool": "fs.read", "arguments": {"path": "x.txt"}, "tool_call_id": "call-1"}}},
                    {"role": "tool", "name": "fs.read", "tool_call_id": "call-1", "content": {"success": True, "result": {"content": "x"}}},
                ],
                [{"name": "fs.read", "description": "read", "parameters": {"type": "object"}}],
                {},
            )
            messages = server.last_request["messages"]  # type: ignore[attr-defined]
            self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-1")
            self.assertEqual(messages[2]["tool_call_id"], "call-1")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_provider_errors_are_typed_and_do_not_echo_keys(self) -> None:
        for mode, kind in (("rate", "rate_limit"), ("auth", "authentication"), ("malformed", "malformed_response")):
            server, thread = self._server(mode)
            try:
                secret = "super-secret-token"
                adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model", secret)
                with self.assertRaises(ProviderError) as context:
                    adapter.generate([], [], {})
                self.assertEqual(context.exception.kind, kind)
                self.assertNotIn(secret, str(context.exception))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=1)

    def test_empty_provider_content_is_retryable(self) -> None:
        original_mode = FakeProviderHandler.mode
        original_do_post = FakeProviderHandler.do_POST
        server, thread = self._server("success")
        try:
            def empty(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": None}}]}).encode("utf-8"))

            FakeProviderHandler.do_POST = empty
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model")
            with self.assertRaises(ProviderError) as context:
                adapter.generate([], [], {})
            self.assertEqual(context.exception.kind, "empty_response")
            self.assertTrue(context.exception.retryable)
        finally:
            FakeProviderHandler.do_POST = original_do_post
            FakeProviderHandler.mode = original_mode
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_timeout_is_retryable(self) -> None:
        server, thread = self._server("timeout")
        try:
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model", timeout_seconds=0.01)
            with self.assertRaises(ProviderError) as context:
                adapter.generate([], [], {})
            self.assertIn(context.exception.kind, {"timeout", "network"})
            self.assertTrue(context.exception.retryable)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_environment_configuration_does_not_require_secret_in_chat_or_code(self) -> None:
        old = dict(os.environ)
        try:
            os.environ["DIVY_MODEL_BASE_URL"] = "http://example.invalid"
            os.environ["DIVY_MODEL_ID"] = "configured-model"
            os.environ["DIVY_MODEL_TIMEOUT_SECONDS"] = "2.5"
            adapter = HttpModelAdapter.from_environment()
            self.assertEqual(adapter.model_id, "configured-model")
            self.assertEqual(adapter.timeout_seconds, 2.5)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_http_error_is_safely_classified_without_provider_body(self) -> None:
        original_mode = FakeProviderHandler.mode
        original_do_post = FakeProviderHandler.do_POST
        server, thread = self._server("success")
        try:
            def rejected(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(413)
                self.end_headers()
                self.wfile.write(b"secret provider payload")

            FakeProviderHandler.do_POST = rejected
            adapter = HttpModelAdapter(f"http://127.0.0.1:{server.server_port}", "test-model")
            with self.assertRaises(ProviderError) as context:
                adapter.generate([], [], {})
            self.assertEqual(context.exception.kind, "request_rejected")
            self.assertNotIn("secret provider payload", str(context.exception))
        finally:
            FakeProviderHandler.do_POST = original_do_post
            FakeProviderHandler.mode = original_mode
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


class PerformanceMemoryTests(unittest.TestCase):
    def test_performance_memory_is_empirical_and_reloadable(self) -> None:
        path = Path("results") / "test-performance-memory.jsonl"
        path.unlink(missing_ok=True)
        try:
            memory = PerformanceMemory(path)
            memory.record(PerformanceObservation("m1", "coding", ("coding",), True, True, True, False, 10, 0.2, 12))
            memory.record(PerformanceObservation("m1", "coding", ("coding",), False, False, True, False, 20, 0.2, 8))
            summary = memory.summarize(model_id="m1", task_type="coding")
            self.assertEqual(summary["observation_count"], 2)
            self.assertEqual(summary["provider_success_rate"], 0.5)
            reloaded = PerformanceMemory(path)
            self.assertEqual(len(reloaded.observations()), 2)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
