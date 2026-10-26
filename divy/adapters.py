"""Provider-neutral HTTP adapter for OpenAI-compatible structured generation.

This adapter is intentionally generic. It does not import or name a vendor;
the endpoint, model identifier, and API key are supplied by configuration.
The existing ``ModelAdapter`` contract is preserved.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .schemas import ModelResponse


@dataclass
class ProviderError(RuntimeError):
    kind: str
    message: str
    status_code: int | None = None
    retryable: bool = False
    retry_after_seconds: float | None = None

    def __str__(self) -> str:
        suffix = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        return f"{self.kind}{suffix}: {self.message}"


class HttpModelAdapter:
    """Call a configured OpenAI-compatible chat-completions endpoint.

    The adapter only performs model I/O. It never executes tools. Tool schemas
    are sent as request metadata and tool execution remains in DIVY's registry.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 30.0,
        endpoint_path: str = "/chat/completions",
        user_agent: str = "divy-ai/0.2",
        response_mode: str = "json_object",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.endpoint_path = endpoint_path
        self.user_agent = user_agent
        self._provider_tool_names: dict[str, str] = {}
        if response_mode not in {"json_object", "json_schema", "none"}:
            raise ValueError("response_mode must be json_object, json_schema, or none")
        self.response_mode = response_mode

    @classmethod
    def from_environment(cls, prefix: str = "DIVY_MODEL_") -> "HttpModelAdapter":
        base_url = os.environ.get(f"{prefix}BASE_URL")
        model_id = os.environ.get(f"{prefix}ID")
        if not base_url or not model_id:
            raise ProviderError("configuration", f"{prefix}BASE_URL and {prefix}ID are required")
        timeout_text = os.environ.get(f"{prefix}TIMEOUT_SECONDS", "30")
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise ProviderError("configuration", f"{prefix}TIMEOUT_SECONDS must be numeric") from exc
        return cls(
            base_url=base_url,
            model_id=model_id,
            api_key=os.environ.get(f"{prefix}API_KEY"),
            timeout_seconds=timeout_seconds,
            endpoint_path=os.environ.get(f"{prefix}ENDPOINT_PATH", "/chat/completions"),
            response_mode=os.environ.get(f"{prefix}RESPONSE_MODE", "json_object"),
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_schema: dict[str, Any],
    ) -> ModelResponse:
        if any(tool.get("x-divy-requires-process-isolation") for tool in tools):
            raise ProviderError("unsafe_execution", "registered test tools require an isolated code runner before hosted-model execution")
        normalized_tools = self._normalize_tools(tools) if tools else []
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._normalize_messages(messages),
            # Some OpenAI-compatible providers default to SSE. DIVY requires a
            # complete response so it can normalize and verify it safely.
            "stream": False,
        }
        if normalized_tools:
            payload["tools"] = normalized_tools
        # Several OpenAI-compatible endpoints reject response_format together
        # with function tools, or silently suppress tool calls. Tool calls are
        # already structurally constrained by the tool schema; use structured
        # response formatting only on tool-free turns.
        if not tools and self.response_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif not tools and self.response_mode == "json_schema" and response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "divy_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}{self.endpoint_path}", data=body, headers=headers, method="POST")
        raw = ""
        content_type = ""
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                self._raise_http_error(exc)
            finally:
                exc.close()
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderError("timeout", "provider request timed out", retryable=True) from exc
        except urllib.error.URLError as exc:
            raise ProviderError("network", "provider connection failed", retryable=True) from None
        except OSError as exc:
            raise ProviderError("network", "provider connection failed", retryable=True) from None

        if content_type.lower().startswith("text/event-stream"):
            raise ProviderError("streaming_unsupported", "streaming responses are not enabled for this adapter")
        response_json = self._parse_json(raw)
        content = self._extract_content(response_json)
        usage = self._extract_usage(response_json)
        return ModelResponse(content=content, token_usage=usage or None)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError("malformed_response", "provider returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ProviderError("malformed_response", "provider response must be a JSON object")
        return parsed

    def _extract_content(self, response_json: dict[str, Any]) -> dict[str, Any]:
        candidate: Any = response_json.get("content")
        tool_calls: Any = None
        if candidate is None:
            choices = response_json.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    candidate = message.get("content")
                    tool_calls = message.get("tool_calls")
        if candidate is None and isinstance(response_json.get("output"), dict):
            candidate = response_json["output"]
        if isinstance(candidate, dict):
            return candidate
        if isinstance(tool_calls, list) and tool_calls:
            normalized_calls: list[dict[str, Any]] = []
            for raw_call in tool_calls:
                if not isinstance(raw_call, dict):
                    raise ProviderError("malformed_response", "tool call must be an object")
                function = raw_call.get("function", {})
                if not isinstance(function, dict):
                    raise ProviderError("malformed_response", "tool call function must be an object")
                name = function.get("name")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ProviderError("malformed_response", "tool-call arguments were not valid JSON") from exc
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise ProviderError("malformed_response", "tool call did not contain a valid function name and object arguments")
                divy_name = self._resolve_provider_tool_name(name)
                # Some hosted models emit a pseudo-tool named ``final`` even
                # when no such tool was registered. Treating it as a final
                # candidate is side-effect free; verification still decides
                # whether the task actually succeeded.
                if divy_name == "final":
                    if len(tool_calls) != 1:
                        raise ProviderError("malformed_response", "final cannot be mixed with tool actions")
                    return {"final": arguments}
                normalized_call = {"tool": divy_name, "arguments": arguments}
                if raw_call.get("id") is not None:
                    normalized_call["tool_call_id"] = raw_call["id"]
                normalized_calls.append(normalized_call)
            if normalized_calls:
                content: dict[str, Any] = {"tool_calls": normalized_calls}
                if len(normalized_calls) == 1:
                    content["action"] = normalized_calls[0]
                return content
        if isinstance(candidate, str):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                # Tool-capable endpoints may disable response_format on turns
                # that include tools and return a plain-text final message.
                # Preserve that message as a final candidate; independent
                # verification remains authoritative and no tool is executed
                # from this fallback.
                if candidate.strip():
                    return {"final": {"message": candidate}}
                raise ProviderError("malformed_response", "model content was not valid JSON") from exc
            if isinstance(parsed, dict):
                return parsed
        raise ProviderError("empty_response", "provider response did not contain content or tool calls", retryable=True)

    @staticmethod
    def _extract_usage(response_json: dict[str, Any]) -> dict[str, int]:
        usage = response_json.get("usage")
        if not isinstance(usage, dict):
            return {}
        result: dict[str, int] = {}
        for source, target in (
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, int):
                result[target] = value
        return result

    @staticmethod
    def _raise_http_error(error: urllib.error.HTTPError) -> None:
        if error.code in {401, 403}:
            raise ProviderError("authentication", "provider rejected authentication", error.code, retryable=False) from error
        if error.code == 429:
            raise ProviderError("rate_limit", "provider rate limit reached", error.code, retryable=True) from error
        if error.code in {408, 409} or 500 <= error.code <= 599:
            raise ProviderError("provider_unavailable", "provider request failed temporarily", error.code, retryable=True) from error
        if error.code in {400, 413, 422}:
            raise ProviderError("request_rejected", "provider rejected the request or context", error.code, retryable=False) from error
        raise ProviderError("provider_error", "provider rejected the request", error.code, retryable=False) from error

    def _normalize_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert DIVY schemas to standard function-tool request objects.

        Already-normalized OpenAI-compatible tool objects pass through. DIVY's
        internal public schema is intentionally simpler: name, description,
        parameters.
        """
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                function = dict(tool["function"])
                name = function.get("name")
                if not isinstance(name, str):
                    raise ProviderError("configuration", "tool schema function must contain a name")
                provider_name = self._provider_tool_name(name)
                function["name"] = provider_name
                normalized.append({**tool, "function": function})
                continue
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(name, str) or not isinstance(parameters, dict):
                raise ProviderError("configuration", "tool schema must contain a name and object parameters")
            provider_name = self._provider_tool_name(name)
            normalized.append({
                "type": "function",
                "function": {
                    "name": provider_name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        return normalized

    def _resolve_provider_tool_name(self, name: str) -> str:
        direct = self._provider_tool_names.get(name)
        if direct is not None:
            return direct
        # Hosted models occasionally detokenize the provider-safe separator
        # ``_x2e_`` as ``.x2e_``. Accept only this deterministic alias and
        # resolve it against the names registered for this request.
        alias = name.replace(".x2e_", "_x2e_")
        return self._provider_tool_names.get(alias, name)

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert DIVY's structured conversation records to chat messages."""
        normalized: list[dict[str, Any]] = []
        generated_call_index = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role == "assistant" and isinstance(content, dict):
                action = content.get("action")
                actions = content.get("tool_calls")
                if isinstance(action, dict):
                    actions = [action]
                if isinstance(actions, list) and actions:
                    provider_calls = []
                    for item in actions:
                        if not isinstance(item, dict):
                            continue
                        tool = item.get("tool")
                        arguments = item.get("arguments", {})
                        if not isinstance(tool, str) or not isinstance(arguments, dict):
                            continue
                        generated_call_index += 1
                        call_id = item.get("tool_call_id") or f"divy-call-{generated_call_index}"
                        provider_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {"name": self._provider_tool_name(tool), "arguments": json.dumps(arguments, sort_keys=True)},
                        })
                    if provider_calls:
                        normalized.append({"role": "assistant", "content": None, "tool_calls": provider_calls})
                        continue
                content = json.dumps(content, sort_keys=True)
            elif role == "tool":
                tool_id = message.get("tool_call_id") or f"divy-call-{max(generated_call_index, 1)}"
                normalized.append({"role": "tool", "tool_call_id": tool_id, "content": json.dumps(content, sort_keys=True) if not isinstance(content, str) else content})
                continue
            elif isinstance(content, (dict, list)):
                content = json.dumps(content, sort_keys=True)
            normalized.append({key: value for key, value in {"role": role, "content": content, "name": message.get("name")}.items() if value is not None})
        return normalized

    def _provider_tool_name(self, name: str) -> str:
        """Map DIVY tool names to provider-safe function identifiers.

        NVIDIA's OpenAI-compatible endpoint rejects dots in function names,
        while DIVY intentionally uses names such as ``fs.read``. The mapping
        stays inside the adapter and is reversed before the response reaches
        AgentSession, so the core ToolRegistry remains unchanged.
        """

        if re.fullmatch(r"[A-Za-z0-9_-]+", name or "") and len(name) <= 64:
            provider_name = name
        else:
            provider_name = re.sub(r"[^A-Za-z0-9_-]", lambda match: f"_x{ord(match.group(0)):02x}_", name)
            if len(provider_name) > 64:
                digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
                provider_name = provider_name[:55] + "_" + digest
        if provider_name in self._provider_tool_names and self._provider_tool_names[provider_name] != name:
            raise ProviderError("configuration", "tool names collide after transport normalization")
        self._provider_tool_names[provider_name] = name
        return provider_name
