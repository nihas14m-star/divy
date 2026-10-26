"""B01-B20 task definitions and deterministic fixture creation."""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import TaskDefinition
from ..tools.workspace import SandboxWorkspace


TASKS = (
    TaskDefinition("B01", "Read/extract", "Read config.json and report its timeout and mode values.", "B01", ("timeout equals 30", "mode equals safe"), ("fs.read",)),
    TaskDefinition("B02", "Search", "Find every TODO and FIXME marker in the fixture workspace.", "B02", ("all expected markers are returned",), ("fs.search",)),
    TaskDefinition("B03", "Structured write", "Set feature_enabled to true in config.json without changing unrelated fields.", "B03", ("feature_enabled equals true", "unrelated fields are preserved"), ("fs.read", "fs.write")),
    TaskDefinition("B04", "Code repair", "Repair calculator.py so add(2, 3) returns 5, then run its registered test.", "B04", ("registered test passes",), ("fs.read", "fs.patch", "test.run")),
    TaskDefinition("B05", "Artifact creation", "Read the two notes and create report.md with Summary, Findings, and Next Steps headings.", "B05", ("report.md exists", "all required headings exist", "source facts are included"), ("fs.read", "fs.write")),
    TaskDefinition("B06", "Computation", "Calculate the total, mean, and maximum amount in data/sales.csv.", "B06", ("total is 100", "mean is 25", "maximum is 40"), ("fs.read", "calc.evaluate")),
    TaskDefinition("B07", "Parallel inspection", "Inspect the three independent fixture files and summarize their roles.", "B07", ("all three files are read", "all roles are identified"), ("fs.read",)),
    TaskDefinition("B08", "Planning", "Plan the steps needed to update a JSON setting and verify it without executing any action.", "B08", ("plan contains read, write, and verify steps",), ()),
    TaskDefinition("B09", "Tool selection", "Calculate 17 times 6 using the calculator tool.", "B09", ("result is 102",), ("calc.evaluate",)),
    TaskDefinition("B10", "No-op recognition", "Ensure config.json already has feature_enabled set to true without changing it.", "B10", ("configuration remains unchanged", "no write occurs"), ("fs.read",)),
    TaskDefinition("B11", "Wrong initial path", "Read the timeout from the expected config path, recovering if that path is wrong.", "B11", ("timeout equals 45", "correct discovered path is used"), ("fs.read", "fs.search")),
    TaskDefinition("B12", "Failed implementation", "Repair transform.py so transform(3, 4) returns 12 and pass its registered test.", "B12", ("registered test passes", "a failed first implementation is recovered"), ("fs.read", "fs.patch", "test.run")),
    TaskDefinition("B13", "Corrupted output", "Create report.md from source.md with the required heading and fact.", "B13", ("report is exact", "corrupted first output is repaired"), ("fs.read", "fs.write")),
    TaskDefinition("B14", "Stale state", "Set enabled to true in config.json while preserving the latest version.", "B14", ("enabled is true", "latest version is preserved", "stale write is recovered"), ("fs.read", "fs.write")),
    TaskDefinition("B15", "Permission recovery", "Create the requested report using the allowed output location if the preferred location is protected.", "B15", ("allowed report exists", "protected location is not bypassed"), ("fs.read", "fs.write")),
    TaskDefinition("B16", "Ambiguity", "Read the configuration file and report its owner when exactly one file is intended.", "B16", ("clarification is requested",), ("fs.list", "fs.search")),
    TaskDefinition("B17", "Stop behavior", "Ensure result.txt contains READY and stop immediately once verified.", "B17", ("result is verified", "no redundant action occurs"), ("fs.read",)),
    TaskDefinition("B18", "Multi-file change", "Rename OLD_KEY to NEW_KEY across source, configuration, and tests, then run the registered test.", "B18", ("all references use NEW_KEY", "registered test passes"), ("fs.search", "fs.patch", "test.run")),
    TaskDefinition("B19", "Delegation", "Inspect inputs/a.txt and inputs/b.txt as independent jobs and combine their values.", "B19", ("both inputs are inspected", "combined summary is correct"), ("fs.read",)),
    TaskDefinition("B20", "End-to-end", "Inspect the project, repair the pipeline, enable the feature, run the test, and write an evidence report.", "B20", ("pipeline test passes", "feature is enabled", "report contains evidence"), ("fs.read", "fs.patch", "fs.write", "test.run")),
)


def all_tasks() -> list[TaskDefinition]:
    return list(TASKS)


def create_fixture(task: TaskDefinition, root: Path) -> SandboxWorkspace:
    protected = {"preferred/report.md"} if task.task_id == "B15" else set()
    workspace = SandboxWorkspace(root, protected_paths=protected)
    task_id = task.task_id
    if task_id == "B01":
        workspace.write_text("config.json", json.dumps({"timeout": 30, "mode": "safe", "retries": 2}, indent=2) + "\n")
    elif task_id == "B02":
        workspace.write_text("src/app.py", "# TODO: replace this placeholder\nprint('app')\n")
        workspace.write_text("docs/notes.md", "FIXME: document the API\n")
        workspace.write_text("README.md", "This fixture has two markers.\n")
    elif task_id == "B03":
        workspace.write_text("config.json", json.dumps({"feature_enabled": False, "name": "divy", "retries": 3}, indent=2) + "\n")
    elif task_id == "B04":
        workspace.write_text("calculator.py", "def add(a, b):\n    return a - b\n")
    elif task_id == "B05":
        workspace.write_text("notes/summary.md", "DIVY is a small orchestration harness.\n")
        workspace.write_text("notes/details.md", "The first milestone measures safe tool use and verification.\n")
    elif task_id == "B06":
        workspace.write_text("data/sales.csv", "item,amount\na,10\nb,20\nc,30\nd,40\n")
    elif task_id == "B07":
        workspace.write_text("config.txt", "Configuration: runtime settings.\n")
        workspace.write_text("pipeline.py", "Pipeline: transforms input data.\n")
        workspace.write_text("README.md", "README: explains how to run the fixture.\n")
    elif task_id == "B09":
        pass
    elif task_id == "B10":
        workspace.write_text("config.json", json.dumps({"feature_enabled": True, "name": "divy", "retries": 3}, indent=2) + "\n")
    elif task_id == "B11":
        workspace.write_text("settings/config.json", json.dumps({"timeout": 45, "mode": "safe"}, indent=2) + "\n")
    elif task_id == "B12":
        workspace.write_text("transform.py", "def transform(a, b):\n    return a - b\n")
    elif task_id == "B13":
        workspace.write_text("source.md", "DIVY verifies artifacts independently.\n")
    elif task_id == "B14":
        workspace.write_text("config.json", json.dumps({"enabled": False, "version": 1}, indent=2) + "\n")
    elif task_id == "B15":
        workspace.write_text("source.md", "Permission recovery must use an explicitly allowed destination.\n")
    elif task_id == "B16":
        workspace.write_text("a/config.json", '{"owner": "alpha"}\n')
        workspace.write_text("b/config.json", '{"owner": "beta"}\n')
    elif task_id == "B17":
        workspace.write_text("result.txt", "READY\n")
    elif task_id == "B18":
        workspace.write_text("src/app.py", "OLD_KEY = 'source'\n")
        workspace.write_text("config.json", '{"OLD_KEY": true}\n')
        workspace.write_text("tests/test_app.py", "assert 'OLD_KEY' in config\n")
    elif task_id == "B19":
        workspace.write_text("inputs/a.txt", "alpha=10\n")
        workspace.write_text("inputs/b.txt", "beta=20\n")
    elif task_id == "B20":
        workspace.write_text("config.json", '{"feature_enabled": false}\n')
        workspace.write_text("pipeline.py", "def run(a, b):\n    return a - b\n")
        workspace.write_text("notes.md", "The pipeline should add its two inputs.\n")
    return workspace


def registered_tests() -> dict:
    from ..tools.tests import load_python_module

    def b04_calculator(workspace: SandboxWorkspace) -> dict:
        module = load_python_module(workspace, "calculator.py")
        actual = module.add(2, 3)
        return {"test_id": "b04_calculator", "passed": actual == 5, "expected": 5, "actual": actual}

    def b12_transform(workspace: SandboxWorkspace) -> dict:
        module = load_python_module(workspace, "transform.py")
        actual = module.transform(3, 4)
        return {"test_id": "b12_transform", "passed": actual == 12, "expected": 12, "actual": actual}

    def b18_keys(workspace: SandboxWorkspace) -> dict:
        paths = ["src/app.py", "config.json", "tests/test_app.py"]
        contents = [workspace.read_text(path) for path in paths]
        passed = all("NEW_KEY" in content and "OLD_KEY" not in content for content in contents)
        return {"test_id": "b18_keys", "passed": passed, "paths": paths}

    def b20_pipeline(workspace: SandboxWorkspace) -> dict:
        module = load_python_module(workspace, "pipeline.py")
        actual = module.run(5, 7)
        return {"test_id": "b20_pipeline", "passed": actual == 12, "expected": 12, "actual": actual}

    return {
        "b04_calculator": b04_calculator,
        "b12_transform": b12_transform,
        "b18_keys": b18_keys,
        "b20_pipeline": b20_pipeline,
    }
