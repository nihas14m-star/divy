from __future__ import annotations

import os
import unittest

from divy.adapters import HttpModelAdapter
from divy.nvidia import NVIDIA_API_BASE_URL, NVIDIA_MODEL_SPECS, nvidia_provider_from_environment, nvidia_registry_from_environment


class NvidiaProviderTests(unittest.TestCase):
    def test_catalog_portfolio_is_provider_local_and_unknown_estimates_stay_unknown(self) -> None:
        old = dict(os.environ)
        try:
            os.environ.pop("DIVY_NVIDIA_MODEL_ID", None)
            os.environ.pop("NVIDIA_MODEL_ID", None)
            provider = nvidia_provider_from_environment()
            self.assertEqual(provider.metadata.provider, "nvidia")
            self.assertEqual(provider.metadata.model_id, "nvidia/nemotron-3.5-bolt-e4b")
            self.assertTrue(provider.metadata.tool_calling)
            self.assertTrue(provider.metadata.coding)
            self.assertTrue(provider.metadata.agentic)
            self.assertIsNone(provider.metadata.cost_estimate)
            self.assertIsNone(provider.metadata.latency_estimate_ms)
            self.assertIsInstance(provider.adapter, HttpModelAdapter)
            self.assertEqual(provider.adapter.base_url, NVIDIA_API_BASE_URL)  # type: ignore[union-attr]
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_model_override_and_portfolio_registry(self) -> None:
        old = dict(os.environ)
        try:
            os.environ["DIVY_NVIDIA_MODEL_ID"] = "nvidia/nemotron-3.5-lightning-30b-a3b"
            provider = nvidia_provider_from_environment()
            self.assertEqual(provider.model_id, "nvidia/nemotron-3.5-lightning-30b-a3b")
            registry = nvidia_registry_from_environment([spec.model_id for spec in NVIDIA_MODEL_SPECS])
            self.assertEqual({item.model_id for item in registry.list()}, {spec.model_id for spec in NVIDIA_MODEL_SPECS})
            self.assertEqual(registry.candidates({"vision"})[0].model_id, "nvidia/nemotron-3.5-super-vl-120b-a12b")
            self.assertEqual(registry.candidates({"tool_calling", "structured_output"})[0].model_id, "nvidia/nemotron-3.5-bolt-e4b")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_unknown_model_is_a_structured_configuration_error(self) -> None:
        old = dict(os.environ)
        try:
            os.environ["DIVY_NVIDIA_MODEL_ID"] = "nvidia/not-a-real-model"
            with self.assertRaisesRegex(Exception, "unsupported NVIDIA model id"):
                nvidia_provider_from_environment()
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_missing_api_key_fails_before_network_access(self) -> None:
        old = dict(os.environ)
        try:
            os.environ.pop("DIVY_NVIDIA_API_KEY", None)
            os.environ.pop("NVIDIA_API_KEY", None)
            os.environ.pop("DIVY_MODEL_API_KEY", None)
            with self.assertRaisesRegex(Exception, "API_KEY.*required"):
                nvidia_provider_from_environment()
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    unittest.main()
