from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from divy.smoke import run_live_smoke


class LiveSmokeTests(unittest.TestCase):
    def test_missing_configuration_is_reported_without_network_call(self) -> None:
        old = dict(os.environ)
        try:
            for name in ("DIVY_MODEL_BASE_URL", "DIVY_MODEL_ID", "DIVY_MODEL_API_KEY"):
                os.environ.pop(name, None)
            with tempfile.TemporaryDirectory() as directory:
                result = run_live_smoke(Path(directory))
            self.assertFalse(result["configured"])
            self.assertFalse(result["success"])
            self.assertEqual(result["reason"], "missing_or_invalid_configuration")
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    unittest.main()
