#!/usr/bin/env python
# Copyright 2025 FlashVLA team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Give a FlashVLA PI0 / PI0.5 checkpoint a config stock lerobot 0.5.1 can read::

    python -m flashvla.policies.export_lerobot_checkpoint <src pretrained_model> <dst_dir>

The parameter tree is already lerobot's, so ``model.safetensors`` is copied
unchanged. What lerobot cannot read is FlashVLA's ``config.json`` (its own
``type`` and extra fields), so ``dst_dir`` gets a ``config.json`` of the matching
lerobot type (``pi05`` or ``pi0``) holding the fields the two configs share,
plus copies of the other files (processors, README). The tensor names and shapes
are checked against lerobot's own model built on the meta device.

A streaming (``*-flashvla``) checkpoint exports as a plain lerobot policy: the
weights load, but they were trained under FlashVLA's buffered action schedule,
so lerobot's synchronous denoising does not reproduce FlashVLA's behavior. PI0
checkpoints trained with ``use_adarms_time_cond`` cannot be exported because
lerobot's PI0 has no adaRMS expert; the check names the offending keys.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path

import draccus
import torch
from safetensors.torch import load_file

LEROBOT_TYPES = {"pi05": "pi05", "pi05-flashvla": "pi05", "pi0": "pi0", "pi0-flashvla": "pi0"}


def _lerobot_classes(lerobot_type: str):
    if lerobot_type == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Config, PI05Policy
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    return PI0Config, PI0Policy


def lerobot_config_dict(flashvla_config: dict) -> dict:
    """Keep the fields lerobot's config of the matching type declares."""
    lerobot_type = LEROBOT_TYPES[flashvla_config["type"]]
    config_cls, _ = _lerobot_classes(lerobot_type)
    fields = {field.name for field in dataclasses.fields(config_cls)} - {"pretrained_path"}
    config = {key: value for key, value in flashvla_config.items() if key in fields}
    config["type"] = lerobot_type
    return config


@contextlib.contextmanager
def _lerobot_choice(lerobot_type: str, config_cls):
    """Point the shared choice registry at lerobot's class while parsing.

    flashvla registers its own ``pi0`` / ``pi05`` config classes under the same
    names, and draccus can only decode a concrete choice class it finds there.
    """
    from lerobot.configs.policies import PreTrainedConfig

    registry = PreTrainedConfig._choice_registry
    previous = registry.get(lerobot_type)
    registry[lerobot_type] = config_cls
    try:
        yield
    finally:
        if previous is None:
            registry.pop(lerobot_type, None)
        else:
            registry[lerobot_type] = previous


def expected_lerobot_shapes(config_file: Path, lerobot_type: str) -> dict[str, torch.Size]:
    """Names and shapes lerobot's policy loads strictly from ``config_file``.

    Parsed the way lerobot's ``PreTrainedConfig.from_pretrained`` does it: the
    concrete class is named explicitly and ``type`` is removed first.
    """
    config_cls, policy_cls = _lerobot_classes(lerobot_type)
    config = json.loads(Path(config_file).read_text())
    config.pop("type", None)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(config, handle)
    try:
        with _lerobot_choice(lerobot_type, config_cls), draccus.config_type("json"):
            parsed = draccus.parse(config_cls, handle.name, args=[])
    finally:
        os.unlink(handle.name)
    parsed.device = "meta"  # lerobot's __init__ moves the model to config.device
    with torch.device("meta"):
        policy = policy_cls(parsed)
    return {name: tensor.shape for name, tensor in policy.state_dict().items()}


def check_against_lerobot(state_dict: dict[str, torch.Tensor], expected: dict[str, torch.Size]) -> None:
    problems = [f"missing {name}" for name in sorted(expected.keys() - state_dict.keys())]
    problems += [f"unexpected {name}" for name in sorted(state_dict.keys() - expected.keys())]
    problems += [
        f"shape {name}: file {tuple(state_dict[name].shape)} vs lerobot {tuple(expected[name])}"
        for name in sorted(expected.keys() & state_dict.keys())
        if state_dict[name].shape != expected[name]
    ]
    if problems:
        raise ValueError("exported checkpoint does not match lerobot's model:\n  " + "\n  ".join(problems))


def export_pretrained_dir(src: Path, dst: Path) -> None:
    flashvla_config = json.loads((src / "config.json").read_text())
    if flashvla_config.get("type") not in LEROBOT_TYPES:
        raise ValueError(f"{src}: type {flashvla_config.get('type')!r} has no lerobot counterpart")
    if src.resolve() == dst.resolve():
        raise ValueError("dst must differ from src; the export is a second copy with lerobot's config")

    config = lerobot_config_dict(flashvla_config)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(config, indent=4) + "\n")

    state_dict = load_file(str(src / "model.safetensors"))
    check_against_lerobot(state_dict, expected_lerobot_shapes(dst / "config.json", config["type"]))

    for item in src.iterdir():
        if item.name == "config.json":
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)
    print(f"wrote {dst}: lerobot type {config['type']}, {len(state_dict)} tensors copied unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="flashvla pretrained_model directory (config.json + model.safetensors)")
    parser.add_argument("dst", type=Path, help="output directory for the lerobot-layout copy")
    args = parser.parse_args()
    export_pretrained_dir(args.src, args.dst)


if __name__ == "__main__":
    main()
