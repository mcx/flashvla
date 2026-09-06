"""The lerobot-layout export matches what stock lerobot 0.5.1 builds, name for name.

Both lerobot policies are built on the meta device from a config derived the way
the exporter derives it, so this checks the key sets and shapes without weights.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from flashvla.policies.export_lerobot_checkpoint import (
    check_against_lerobot,
    expected_lerobot_shapes,
    lerobot_config_dict,
)
from flashvla.policies.loading import to_lerobot_state_dict
from flashvla.policies.pi0.configuration_pi0 import PI0Config, PI0FlashVLAConfig
from flashvla.policies.pi0.modeling_pi0 import PI0Policy
from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy
from flashvla.policies.pi05.configuration_pi05 import PI05Config, PI05FlashVLAConfig
from flashvla.policies.pi05.modeling_pi05 import PI05Policy
from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy


def _config(config_cls, **overrides):
    config = config_cls(**overrides)
    config.input_features = {
        f"observation.images.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        for cam in ("a", "b")
    }
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    config.device = "cpu"
    config.compile_model = False
    return config


class LerobotExportTest(unittest.TestCase):
    def _exported(self, policy_cls, config):
        with torch.device("meta"):
            policy = policy_cls(config)
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)  # flashvla's config.json, as a checkpoint carries it
            flashvla_config = json.loads((Path(tmp) / "config.json").read_text())
            lerobot_config = lerobot_config_dict(flashvla_config)
            (Path(tmp) / "lerobot.json").write_text(json.dumps(lerobot_config))
            expected = expected_lerobot_shapes(Path(tmp) / "lerobot.json", lerobot_config["type"])
        exported = to_lerobot_state_dict({k: v for k, v in policy.state_dict().items()})
        return lerobot_config, exported, expected

    def test_pi05_exports_match_lerobot(self) -> None:
        for policy_cls, config_cls in ((PI05Policy, PI05Config), (PI05FlashVLAPolicy, PI05FlashVLAConfig)):
            with self.subTest(policy_cls.__name__):
                lerobot_config, exported, expected = self._exported(policy_cls, _config(config_cls))
                self.assertEqual(lerobot_config["type"], "pi05")
                self.assertNotIn("num_buffer_slots", lerobot_config)
                self.assertEqual(lerobot_config["input_features"].keys(), {"observation.images.a", "observation.images.b", "observation.state"})
                check_against_lerobot(exported, expected)  # raises on any name/shape difference
                self.assertEqual(len(exported), 813)

    def test_pi0_exports_match_lerobot(self) -> None:
        for policy_cls, config_cls in ((PI0Policy, PI0Config), (PI0FlashVLAPolicy, PI0FlashVLAConfig)):
            with self.subTest(policy_cls.__name__):
                lerobot_config, exported, expected = self._exported(policy_cls, _config(config_cls))
                self.assertEqual(lerobot_config["type"], "pi0")
                check_against_lerobot(exported, expected)
                self.assertEqual(len(exported), 778)

    def test_adarms_pi0_is_refused(self) -> None:
        _, exported, expected = self._exported(
            PI0FlashVLAPolicy, _config(PI0FlashVLAConfig, use_adarms_time_cond=True)
        )
        with self.assertRaisesRegex(ValueError, "does not match lerobot"):
            check_against_lerobot(exported, expected)


if __name__ == "__main__":
    unittest.main()
