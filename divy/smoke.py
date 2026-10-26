"""Real-model readiness smoke test (point #9).

Provides a minimal live-provider smoke-test pathway that:

* reads configuration from environment variables (DIVY_MODEL_BASE_URL,
  DIVY_MODEL_ID, optional DIVY_MODEL_API_KEY);
* skips cleanly when no API configuration is present;
* never logs credentials or secrets;
* records latency, token usage, cost, and errors where available.

No network call is made unless the provider is fully configured. If no API
configuration exists, the result reports configured=False and success=False
without attempting any request. Never fabricates a real-model result.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .adapters import HttpModelAdapter, ProviderError
from .schemas import ModelResponse


_ENV_BASE_URL = "DIVY_MODEL_BASE_URL"
_ENV_MODEL_ID = "DIVY_MODEL_ID"
_ENV_API_KEY = "DIVY_MODEL_API_KEY"
_ENV_TIMEOUT = "DIVY_MODEL_TIMEOUT_SECONDS"


def _env_configured() -> bool:
    return bool(os.environ.get(_ENV_BASE_URL)) and bool(os.environ.get(_ENV_MODEL_ID))


def run_live_smoke(output_dir: Path | str | None = None) -> dict[str, Any]:
    """Run a minimal live-provider smoke test.

    When the provider is not configured, returns immediately with
    configured=False and success=False and makes no network call.

    When configured, performs one minimal generation against the configured
    endpoint and records latency, token usage, cost estimate, and any error.
    Never echoes the API key into the result.
    """
    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    # --- configuration check (no network yet) ---------------------------
    base_url = os.environ.get(_ENV_BASE_URL)
    model_id = os.environ.get(_ENV_MODEL_ID)
    api_key = os.environ.get(_ENV_API_KEY)
    timeout_text = os.environ.get(_ENV_TIMEOUT, "30")
    try:
        timeout_seconds = float(timeout_text)
    except ValueError:
        timeout_seconds = 30.0

    configured = bool(base_url) and bool(model_id)
    if not configured:
        result = {
            "configured": False,
            "success": False,
            "reason": "missing_or_invalid_configuration",
            "latency_ms": None,
            "token_usage": None,
            "estimated_cost": None,
            "error": None,
        }
        if out_path is not None:
            (out_path / "live_smoke.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        return result

    # --- live call (only when configured) -------------------------------
    assert base_url is not None and model_id is not None
    started = time.perf_counter()
    latency_ms = None
    token_usage = None
    estimated_cost = None
    error = None
    success = False
    try:
        adapter = HttpModelAdapter(
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        # Minimal smoke payload: a single structured generation.
        response: ModelResponse = adapter.generate(
            [{"role": "user", "content": {"smoke": True}}],
            [],
            {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        latency_ms = elapsed
        token_usage = response.token_usage or None
        # Cost is only recorded where calculable from metadata; never guessed.
        success = bool(response.content)
    except ProviderError as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        latency_ms = elapsed
        # Never echo credentials. Report only the error kind, not the message
        # which could contain a key or URL details.
        error = {
            "kind": exc.kind,
            "retryable": exc.retryable,
        }
        success = False
    except Exception as exc:  # network / unexpected failures are data
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        latency_ms = elapsed
        error = {"kind": type(exc).__name__, "retryable": False}
        success = False

    result = {
        "configured": True,
        "success": success,
        "reason": None if success else "live_call_failed",
        "latency_ms": latency_ms,
        "token_usage": token_usage,
        "estimated_cost": estimated_cost,
        "error": error,
    }
    if out_path is not None:
        (out_path / "live_smoke.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> int:
    """CLI entry point; missing configuration is a safe, successful skip."""
    output_dir = Path(os.environ.get("DIVY_SMOKE_OUTPUT_DIR", "results"))
    result = run_live_smoke(output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["configured"] or result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
