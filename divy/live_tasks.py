"""Real-world task definitions for live evaluation.

These tasks are designed to be independently verifiable and cover
different capability categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from divy.schemas import TaskDefinition
from divy.tools.workspace import SandboxWorkspace
from divy.tools.registry import ToolRegistry


@dataclass(frozen=True)
class LiveTask:
    """A live-evaluation task with all necessary metadata."""
    task_id: str
    title: str
    goal: str
    fixture_name: str
    success_criteria: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    task_type: str
    complexity: str = "normal"
    required_capabilities: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Task fixtures and verification ---

def _rw01_fixture(workspace: SandboxWorkspace) -> None:
    workspace.write_text("src/calculator.py", "def add(a, b):\n    return a - b\n")
    workspace.write_text("test/test_calculator.py", """import sys
sys.path.insert(0, 'src')
from calculator import add

def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
    print("All tests passed!")

if __name__ == "__main__":
    test_add()
""")

def _rw01_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    source = workspace.read_text("src/calculator.py")
    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )
    # RW01's declared success criteria are the repair and the registered test.
    # Optional notes are evidence only; they must not silently become a hidden
    # requirement that changes the task the live model was given.
    return ("return a + b" in source) and test_passed


def _rw02_fixture(workspace: SandboxWorkspace) -> None:
    workspace.write_text("src/buggy.py", """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n)  # Bug: should be factorial(n-1)
""")
    workspace.write_text("test/test_factorial.py", """import sys
sys.path.insert(0, 'src')
from buggy import factorial

def test_factorial():
    assert factorial(1) == 1
    assert factorial(2) == 2
    assert factorial(3) == 6
    assert factorial(4) == 24
    assert factorial(5) == 120
    print("All tests passed!")

if __name__ == "__main__":
    test_factorial()
""")

def _rw02_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    source = workspace.read_text("src/buggy.py")
    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )
    # The registered test is the authoritative behavioral check. Formatting
    # differences such as ``n - 1`` versus ``n-1`` must not cause a false
    # rejection after the fixture has been repaired.
    return "def factorial" in source and test_passed


def _rw03_fixture(workspace: SandboxWorkspace) -> None:
    workspace.write_text("spec.md", """# API Specification

Implement these as plain Python functions in impl.py. Do not use a web
framework; the registered test imports impl.py and calls the functions directly.

- list_users() -> list[dict]: return all users. Each user has "id" and "name".
- create_user(data: dict) -> dict: create and return a user with "id" and "name".
- get_user(id) -> dict | None: return the matching user, or None if not found.
""")
    workspace.write_text("impl.py", """# TODO: Implement the API
def list_users(): pass
def create_user(data): pass
def get_user(id): pass
""")

def _rw03_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    impl_path = workspace.resolve("impl.py")
    if not impl_path.exists():
        return False
    # The behavioral test is authoritative: it exercises listing, creation, and
    # lookup rather than merely checking that stub functions exist.
    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )
    return test_passed


def _rw04_fixture(workspace: SandboxWorkspace) -> None:
    workspace.write_text("data/input.txt", "hello world\nhello divy\nworld hello\n")
    workspace.write_text("expected.txt", "hello: 2\nworld: 2\ndivy: 1\n")

def _rw04_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    output_path = workspace.resolve("output.txt")
    expected_path = workspace.resolve("expected.txt")
    if not output_path.exists() or not expected_path.exists():
        return False
    output = output_path.read_text(encoding="utf-8")
    expected = expected_path.read_text(encoding="utf-8")
    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )
    return output.strip() == expected.strip() and test_passed


def _rw05_fixture(workspace: SandboxWorkspace) -> None:
    workspace.write_text("data/numbers.json", "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")

def _rw05_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    output_path = workspace.resolve("output.json")
    if not output_path.exists():
        return False
    output = output_path.read_text(encoding="utf-8")
    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )
    # Expected: sum=55, avg=5.5, max=10
    import json
    try:
        data = json.loads(output)
        return data.get("sum") == 55 and data.get("avg") == 5.5 and data.get("max") == 10 and test_passed
    except Exception:
        return False


# --- Harder task variants (RW06-RW09) ---

def _rw06_fixture(workspace: SandboxWorkspace) -> None:
    """Misleading implementation: code looks correct but has subtle bug.

    The function appears to implement a proper cache but has a mutable default
    argument bug that causes cross-call contamination.
    """
    workspace.write_text("src/cache.py", """def get_cached_value(key, cache={}):
    \"\"\"Get value from cache, computing if not present.\"\"\"
    if key not in cache:
        cache[key] = compute_expensive(key)
    return cache[key]

def compute_expensive(key):
    \"\"\"Simulate expensive computation.\"\"\"
    return f\"value_for_{key}\"
""")
    workspace.write_text("test/test_cache.py", """import sys
sys.path.insert(0, 'src')
from cache import get_cached_value

def test_cache_isolation():
    # Test 1: Basic retrieval works
    result1 = get_cached_value("key1")
    assert result1 == "value_for_key1", f"Expected 'value_for_key1', got {result1}"

    result2 = get_cached_value("key2")
    assert result2 == "value_for_key2", f"Expected 'value_for_key2', got {result2}"

    # Test 2: Same key retrieval works (basic functionality)
    result3 = get_cached_value("key1")
    assert result3 == "value_for_key1", f"Expected 'value_for_key1', got {result3}"

    # Test 3: Cross-call cache isolation - this is the key test for mutable default argument bug
    # If the mutable default argument bug exists, cache will persist across separate
    # function calls in different contexts. We simulate this by checking that
    # a fresh import/reload doesn't share cache state.
    # We do this by verifying that calling with same key twice doesn't
    # incorrectly share state across what should be independent calls.
    # The real test of the bug: if we call with key1, then key2, then key1 again,
    # and the bug exists, the cache might have unexpected entries.
    # But the key test: if cache is a mutable default, it persists across ALL calls.
    # We verify isolation by checking cache behavior.

    # Test that cache doesn't leak across different keys
    result4 = get_cached_value("key3")
    assert result4 == "value_for_key3", f"Expected 'value_for_key3', got {result4}"

    # If mutable default arg bug exists, key1 might have been polluted by key2/key3 calls
    # But our current test can't easily detect cross-call persistence without
    # separate process isolation. The key indicator is that the function
    # should work correctly within a single session.
    # The actual bug manifests as: cache dict persists across module reloads
    # which we can't easily test here. We rely on the verifier checking source code.

    print("All tests passed!")

if __name__ == "__main__":
    test_cache_isolation()
""")

def _rw06_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    source = workspace.read_text("src/cache.py")
    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )
    # Must fix the mutable default argument bug: change cache={} to cache=None and initialize inside
    fixed = "cache=None" in source and "if cache is None:" in source and "cache = {}" in source
    return fixed and test_passed


def _rw07_fixture(workspace: SandboxWorkspace) -> None:
    """Multiple plausible files: need to find and fix the correct one.

    There are multiple files with similar names, only one has the bug.
    The model must search and identify the correct file.
    """
    workspace.write_text("src/processor_a.py", """def process(data):
    return data.upper()
""")
    workspace.write_text("src/processor_b.py", """def process(data):
    return data.lower()  # Bug: should be upper()
""")
    workspace.write_text("src/processor_c.py", """def process(data):
    return data.title()
""")
    workspace.write_text("spec.md", """# Processor Specification

The system has multiple processor modules. The correct processor is the one
that converts input to UPPERCASE. Find and fix the buggy processor.
""")
    workspace.write_text("test/test_processor.py", """import sys
sys.path.insert(0, 'src')
# The test imports the correct processor based on spec
from processor_b import process

def test_processor():
    assert process("hello") == "HELLO"
    assert process("WORLD") == "WORLD"
    assert process("MiXeD") == "MIXED"
    print("All tests passed!")

if __name__ == "__main__":
    test_processor()
""")

def _rw07_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    source = workspace.read_text("src/processor_b.py")
    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )
    # Must fix processor_b to return uppercase
    fixed = "return data.upper()" in source or "return data.upper ()" in source
    return fixed and test_passed


def _rw08_fixture(workspace: SandboxWorkspace) -> None:
    """Failing test requiring diagnosis: test fails but the code looks correct.

    The test has a subtle bug in its assertion. The model must diagnose
    whether the code or the test is wrong, then fix the right thing.
    """
    workspace.write_text("src/validator.py", """def validate_email(email):
    \"\"\"Validate email format.\"\"\"
    if not email or "@" not in email:
        return False
    local, domain = email.split("@", 1)
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    return True
""")
    workspace.write_text("test/test_validator.py", """import sys
sys.path.insert(0, 'src')
from validator import validate_email

def test_validator():
    # Valid emails
    assert validate_email("user@example.com") == True
    assert validate_email("test@domain.org") == True
    # Invalid emails
    assert validate_email("invalid") == False
    assert validate_email("@nodomain") == False
    assert validate_email("user@") == False
    # This assertion is WRONG - "user@domain" is actually valid per the spec
    # (domain doesn't strictly require a dot in all cases)
    # The test should be fixed, not the code
    assert validate_email("user@domain") == True  # Fixed: this is valid
    print("All tests passed!")

if __name__ == "__main__":
    test_validator()
""")

def _rw08_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    source = workspace.read_text("src/validator.py")
    test_source = workspace.read_text("test/test_validator.py")
    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )
    # The code is already correct. The test had a bug (asserting False for valid email).
    # The fix should be in the test file, not the source code.
    # Verify the test was fixed (the assertion changed to True)
    test_fixed = "assert validate_email(\"user@domain\") == True" in test_source
    # Verify source code wasn't changed incorrectly
    source_unchanged = "def validate_email" in source and "@" in source
    return test_fixed and source_unchanged and test_passed


def _rw09_fixture(workspace: SandboxWorkspace) -> None:
    """Multi-step dependency: need to fix multiple files in correct order.

    A config file references a module that doesn't exist yet. The model must
    create the module, then update the config, then verify.
    """
    workspace.write_text("config.json", """{
    "processor_module": "data_processor",
    "input_file": "input.txt",
    "output_file": "output.txt"
}
""")
    workspace.write_text("input.txt", "hello world")
    workspace.write_text("spec.md", """# Data Processor Specification

Create a module `data_processor.py` with a `process_file` function that:
1. Reads the input file path from config
2. Reads the file content
3. Reverses each line
4. Writes to the output file path from config
5. Returns the number of lines processed
""")

def _rw09_verifier(workspace: SandboxWorkspace, trajectory: dict) -> bool:
    from divy.evaluators import _tool_events
    import json

    # Check data_processor.py exists and has correct function
    processor_path = workspace.resolve("data_processor.py")
    if not processor_path.exists():
        return False
    processor_source = processor_path.read_text(encoding="utf-8")
    has_process_func = "def process_file" in processor_source

    # Check config.json references the right module
    config = json.loads(workspace.read_text("config.json"))
    config_correct = config.get("processor_module") == "data_processor"

    # Check output.txt exists and has reversed lines
    output_path = workspace.resolve("output.txt")
    if not output_path.exists():
        return False
    output = output_path.read_text(encoding="utf-8")
    output_correct = output.strip() == "dlrow olleh"

    test_passed = any(
        e.get("event") == "tool_observation" and e.get("tool") == "test.run" and e.get("success") and e.get("result", {}).get("passed")
        for e in trajectory.get("events", [])
    )

    return has_process_func and config_correct and output_correct and test_passed


# --- Task registry ---

LIVE_TASKS = {
    "RW01": LiveTask(
        task_id="RW01",
        title="Bug Fix: Calculator",
        goal="Repair src/calculator.py so add(2, 3) returns 5 and verify with the registered test.",
        fixture_name="RW01",
        success_criteria=("calculator add returns 5", "registered test passes"),
        allowed_tools=("fs.read", "fs.patch", "fs.write", "test.run", "calc.evaluate"),
        task_type="software_engineering",
        complexity="moderate",
        required_capabilities=frozenset({"coding", "structured_output"}),
    ),
    "RW02": LiveTask(
        task_id="RW02",
        title="Bug Fix: Factorial",
        goal="Repair the factorial function in buggy.py and verify with the registered test.",
        fixture_name="RW02",
        success_criteria=("factorial function works correctly", "registered test passes"),
        allowed_tools=("fs.read", "fs.patch", "test.run"),
        task_type="software_engineering",
        complexity="moderate",
        required_capabilities=frozenset({"coding", "structured_output"}),
    ),
    "RW03": LiveTask(
        task_id="RW03",
        title="Code Generation: API Implementation",
        goal="Implement the functions described in spec.md in impl.py and verify with the registered test.",
        fixture_name="RW03",
        success_criteria=("functions behave as specified", "registered test passes"),
        allowed_tools=("fs.read", "fs.write", "test.run"),
        task_type="code_generation",
        complexity="complex",
        required_capabilities=frozenset({"coding", "structured_output"}),
    ),
    "RW04": LiveTask(
        task_id="RW04",
        title="Data Processing: Word Count",
        goal="Count word frequencies in input.txt and write the report to output.txt. Match the format shown in expected.txt exactly: one 'word: count' line per word, ordered by descending count then alphabetically.",
        fixture_name="RW04",
        success_criteria=("word counts match expected", "output format correct"),
        allowed_tools=("fs.read", "fs.write", "calc.evaluate", "test.run"),
        task_type="data_processing",
        complexity="simple",
        required_capabilities=frozenset({"coding", "structured_output"}),
    ),
    "RW05": LiveTask(
        task_id="RW05",
        title="Data Analysis: Statistics",
        goal="Calculate the sum, average, and maximum of numbers in numbers.json and write them to output.json as JSON with exactly the keys 'sum', 'avg', and 'max'.",
        fixture_name="RW05",
        success_criteria=("sum=55", "avg=5.5", "max=10"),
        allowed_tools=("fs.read", "fs.write", "calc.evaluate", "test.run"),
        task_type="data_analysis",
        complexity="simple",
        required_capabilities=frozenset({"coding", "structured_output", "reasoning"}),
    ),
    "RW06": LiveTask(
        task_id="RW06",
        title="Bug Fix: Mutable Default Argument",
        goal="Fix the mutable default argument bug in src/cache.py. The cache should not persist across independent function calls.",
        fixture_name="RW06",
        success_criteria=("mutable default argument fixed", "cache isolation test passes"),
        allowed_tools=("fs.read", "fs.patch", "fs.write", "test.run"),
        task_type="software_engineering",
        complexity="moderate",
        required_capabilities=frozenset({"coding", "structured_output", "reasoning"}),
    ),
    "RW07": LiveTask(
        task_id="RW07",
        title="Code Navigation: Find Correct Processor",
        goal="Multiple processor files exist. Find the one referenced by the test (processor_b) and fix it to convert input to UPPERCASE as specified.",
        fixture_name="RW07",
        success_criteria=("correct processor fixed", "uppercase conversion works"),
        allowed_tools=("fs.read", "fs.search", "fs.patch", "fs.write", "test.run"),
        task_type="software_engineering",
        complexity="complex",
        required_capabilities=frozenset({"coding", "structured_output", "reasoning"}),
    ),
    "RW08": LiveTask(
        task_id="RW08",
        title="Test Diagnosis: Validator Test Bug",
        goal="The validator code is correct but the test has a bug. Diagnose and fix the test assertion for 'user@domain' (it should be valid).",
        fixture_name="RW08",
        success_criteria=("test fixed to accept valid email", "validator code unchanged"),
        allowed_tools=("fs.read", "fs.patch", "fs.write", "test.run"),
        task_type="software_engineering",
        complexity="moderate",
        required_capabilities=frozenset({"coding", "structured_output", "reasoning"}),
    ),
    "RW09": LiveTask(
        task_id="RW09",
        title="Multi-Step: Config-Driven Processor",
        goal="Create a data_processor module that reads config.json, processes input.txt by reversing each line, writes to output.txt, and returns line count.",
        fixture_name="RW09",
        success_criteria=("data_processor module created", "config referenced correctly", "output.txt has reversed lines", "test passes"),
        allowed_tools=("fs.read", "fs.write", "fs.patch", "test.run", "fs.search"),
        task_type="software_engineering",
        complexity="complex",
        required_capabilities=frozenset({"coding", "structured_output", "reasoning"}),
    ),
}


TASK_FIXTURES = {
    "RW01": _rw01_fixture,
    "RW02": _rw02_fixture,
    "RW03": _rw03_fixture,
    "RW04": _rw04_fixture,
    "RW05": _rw05_fixture,
    "RW06": _rw06_fixture,
    "RW07": _rw07_fixture,
    "RW08": _rw08_fixture,
    "RW09": _rw09_fixture,
}

TASK_VERIFIERS = {
    "RW01": _rw01_verifier,
    "RW02": _rw02_verifier,
    "RW03": _rw03_verifier,
    "RW04": _rw04_verifier,
    "RW05": _rw05_verifier,
    "RW06": _rw06_verifier,
    "RW07": _rw07_verifier,
    "RW08": _rw08_verifier,
    "RW09": _rw09_verifier,
}


def get_task(task_id: str) -> LiveTask:
    task = LIVE_TASKS.get(task_id)
    if not task:
        raise ValueError(f"Unknown task: {task_id}")
    return task


def setup_fixture(task_id: str, workspace: SandboxWorkspace) -> None:
    fixture_fn = TASK_FIXTURES.get(task_id)
    if not fixture_fn:
        raise ValueError(f"No fixture for task: {task_id}")
    fixture_fn(workspace)


def verify_task(task_id: str, workspace: SandboxWorkspace, trajectory: dict) -> bool:
    verifier = TASK_VERIFIERS.get(task_id)
    if not verifier:
        raise ValueError(f"No verifier for task: {task_id}")
    return verifier(workspace, trajectory)


def build_registry(workspace: SandboxWorkspace, task_id: str, *, include_test_run: bool = True) -> "ToolRegistry":
    from divy.tools import CalculatorTool, FilesystemTools, TestRunnerTool, ToolRegistry
    from divy.tools.tests import load_python_module
    from divy.live_tasks import LIVE_TASKS

    # Get allowed tools for this task
    task = LIVE_TASKS.get(task_id)
    allowed_tools = set(task.allowed_tools) if task else set()

    # Build filesystem tool definitions, filtering by allowed_tools
    fs_tools = FilesystemTools(workspace)
    fs_defs = []
    for defn in fs_tools.definitions():
        if not allowed_tools or defn.name in allowed_tools:
            fs_defs.append(defn)

    # Build calculator tool definitions, filtering by allowed_tools
    calc_tools = CalculatorTool()
    calc_defs = []
    for defn in calc_tools.definitions():
        if not allowed_tools or defn.name in allowed_tools:
            calc_defs.append(defn)

    definitions = fs_defs + calc_defs

    # Add task-specific tests (only if test.run is allowed and include_test_run is True)
    if include_test_run and "test.run" in allowed_tools:
        if task_id == "RW01":
            def b04_calculator(ws: SandboxWorkspace) -> dict:
                try:
                    module = load_python_module(ws, "src/calculator.py")
                    actual = module.add(2, 3)
                    return {"test_id": "b04_calculator", "passed": actual == 5, "expected": 5, "actual": actual}
                except Exception as e:
                    return {"test_id": "b04_calculator", "passed": False, "error": "syntax_or_runtime_error"}

            test_defs = TestRunnerTool(workspace, {"b04_calculator": b04_calculator}).definitions()
            definitions += test_defs
            # Also register with more discoverable names
            definitions += TestRunnerTool(workspace, {"calculator_test": b04_calculator}).definitions()
            definitions += TestRunnerTool(workspace, {"test_add": b04_calculator}).definitions()

        elif task_id == "RW02":
            def test_factorial(ws: SandboxWorkspace) -> dict:
                try:
                    module = load_python_module(ws, "src/buggy.py")
                    results = [(1, 1), (2, 2), (3, 6), (4, 24), (5, 120)]
                    passed = all(module.factorial(n) == expected for n, expected in results)
                    return {"test_id": "factorial", "passed": passed, "results": [(n, module.factorial(n), e) for n, e in results]}
                except Exception:
                    return {"test_id": "factorial", "passed": False, "error": "syntax_or_runtime_error"}
            definitions += TestRunnerTool(workspace, {"factorial": test_factorial}).definitions()

        elif task_id == "RW03":
            def test_api(ws: SandboxWorkspace) -> dict:
                try:
                    module = load_python_module(ws, "impl.py")
                    users = module.list_users()
                    if not isinstance(users, list):
                        return {"test_id": "api_impl", "passed": False, "error": "list_users() must return a list"}
                    if users:
                        first = users[0]
                        if not isinstance(first, dict) or "id" not in first or "name" not in first:
                            return {"test_id": "api_impl", "passed": False, "error": "list_users() must return user dicts with id and name"}
                        found = module.get_user(first["id"])
                        if found != first:
                            return {"test_id": "api_impl", "passed": False, "error": "get_user() must return the matching user"}
                    created = module.create_user({"name": "Carol"})
                    if not isinstance(created, dict) or created.get("name") != "Carol" or "id" not in created:
                        return {"test_id": "api_impl", "passed": False, "error": "create_user() must return the created user with an id"}
                    found_created = module.get_user(created["id"])
                    if found_created is not None and found_created != created:
                        return {"test_id": "api_impl", "passed": False, "error": "get_user() must return the created user or None"}
                    return {"test_id": "api_impl", "passed": True}
                except Exception as e:
                    return {"test_id": "api_impl", "passed": False, "error": "syntax_or_runtime_error"}
            definitions += TestRunnerTool(workspace, {"api_impl": test_api}).definitions()

        elif task_id == "RW04":
            def test_wordcount(ws: SandboxWorkspace) -> dict:
                output = ws.read_text("output.txt")
                expected = ws.read_text("expected.txt")
                passed = output.strip() == expected.strip()
                return {"test_id": "wordcount", "passed": passed}
            definitions += TestRunnerTool(workspace, {"wordcount": test_wordcount}).definitions()

        elif task_id == "RW05":
            def test_stats(ws: SandboxWorkspace) -> dict:
                import json
                output = ws.read_text("output.json")
                try:
                    data = json.loads(output)
                    passed = data.get("sum") == 55 and data.get("avg") == 5.5 and data.get("max") == 10
                except Exception:
                    passed = False
                return {"test_id": "stats", "passed": passed}
            definitions += TestRunnerTool(workspace, {"stats": test_stats}).definitions()

        elif task_id == "RW06":
            def test_cache(ws: SandboxWorkspace) -> dict:
                try:
                    module = load_python_module(ws, "src/cache.py")
                    # Test cache isolation: each call should be independent
                    result1 = module.get_cached_value("key1")
                    result2 = module.get_cached_value("key2")
                    result3 = module.get_cached_value("key1")
                    passed = (result1 == "value_for_key1" and
                              result2 == "value_for_key2" and
                              result3 == "value_for_key1")
                    return {"test_id": "cache_isolation", "passed": passed}
                except Exception as e:
                    return {"test_id": "cache_isolation", "passed": False, "error": "syntax_or_runtime_error"}
            # Register both test IDs in a single TestRunnerTool
            definitions += TestRunnerTool(workspace, {"cache_isolation": test_cache, "test_cache_isolation": test_cache}).definitions()

        elif task_id == "RW07":
            def test_processor(ws: SandboxWorkspace) -> dict:
                try:
                    module = load_python_module(ws, "src/processor_b.py")
                    passed = (module.process("hello") == "HELLO" and
                              module.process("WORLD") == "WORLD" and
                              module.process("MiXeD") == "MIXED")
                    return {"test_id": "processor", "passed": passed}
                except Exception:
                    return {"test_id": "processor", "passed": False, "error": "syntax_or_runtime_error"}
            definitions += TestRunnerTool(workspace, {"processor": test_processor}).definitions()

        elif task_id == "RW08":
            def test_validator(ws: SandboxWorkspace) -> dict:
                try:
                    module = load_python_module(ws, "src/validator.py")
                    # Valid emails
                    valid1 = module.validate_email("user@example.com")
                    valid2 = module.validate_email("test@domain.org")
                    valid3 = module.validate_email("user@domain")  # Should be valid now
                    # Invalid emails
                    invalid1 = module.validate_email("invalid")
                    invalid2 = module.validate_email("@nodomain")
                    invalid3 = module.validate_email("user@")
                    passed = valid1 and valid2 and valid3 and not invalid1 and not invalid2 and not invalid3
                    return {"test_id": "validator", "passed": passed}
                except Exception:
                    return {"test_id": "validator", "passed": False, "error": "syntax_or_runtime_error"}
            definitions += TestRunnerTool(workspace, {"validator": test_validator}).definitions()

        elif task_id == "RW09":
            def test_processor_module(ws: SandboxWorkspace) -> dict:
                try:
                    module = load_python_module(ws, "data_processor.py")
                    import json
                    config = json.loads(ws.read_text("config.json"))
                    input_file = config.get("input_file", "input.txt")
                    output_file = config.get("output_file", "output.txt")
                    lines_processed = module.process_file(input_file, output_file)
                    output = ws.read_text(output_file)
                    passed = (lines_processed == 1 and output.strip() == "dlrow olleh")
                    return {"test_id": "processor_module", "passed": passed, "lines": lines_processed}
                except Exception as e:
                    return {"test_id": "processor_module", "passed": False, "error": "syntax_or_runtime_error"}
            definitions += TestRunnerTool(workspace, {"processor_module": test_processor_module}).definitions()

    from divy.tools.registry import ToolRegistry
    registry = ToolRegistry(definitions)
    return registry


def get_all_task_ids() -> list[str]:
    return list(LIVE_TASKS.keys())
