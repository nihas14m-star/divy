"""Read-only artifact consistency/security check; no provider requests."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    secrets = [os.environ[k] for k in ("NVIDIA_API_KEY", "DIVY_NVIDIA_API_KEY", "DIVY_MODEL_API_KEY") if os.environ.get(k)]
    pattern = re.compile(r"nvapi-[A-Za-z0-9_-]{20,}")
    scanned = 0
    credential_files = 0
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".md", ".json", ".jsonl", ".txt"} or not path.is_file():
            continue
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        scanned += 1
        if pattern.search(text) or any(secret in text for secret in secrets):
            credential_files += 1
    counts = defaultdict(lambda: {"trials": 0, "successful": 0, "selected_models": {}})
    multiple_execution_paths = 0
    cold = root / "results" / "live_cold_start" / "trials"
    for path in cold.glob("*/trial_summary.json"):
        trial = json.loads(path.read_text(encoding="utf-8"))
        row = counts[trial["execution_mode"]]
        row["trials"] += 1
        row["successful"] += int(trial["success"])
        model = trial["model_id"]
        row["selected_models"][model] = row["selected_models"].get(model, 0) + 1
        trajectories = path.parent / "trajectory.jsonl"
        if trajectories.exists():
            ids = {json.loads(line).get("execution_id") for line in trajectories.read_text(encoding="utf-8").splitlines() if line.strip()}
            multiple_execution_paths += int(len(ids) > 1)
    print(json.dumps({
        "files_scanned": scanned,
        "credential_matches_files": credential_files,
        "live_cold_start_latest_summaries": dict(counts),
        "trial_paths_with_multiple_execution_ids": multiple_execution_paths,
        "live_requests_made": 0,
    }, indent=2))
    return 1 if credential_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
