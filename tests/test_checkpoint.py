"""Integrity checks for the reference artifact, independent of optional RL packages.

The tests pin the published artifact by SHA-256, verify the archive structure, and check the serialized PPO network configuration.
"""
import hashlib
import json
from pathlib import Path
import zipfile


CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "navigation_ppo.zip"
SHA256 = "f6b44858775154b052c58286beeb713d40fd82bb08767d722f21462de2f16786"


def test_checkpoint_is_valid_zip():
    assert CHECKPOINT.is_file()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == SHA256
    with zipfile.ZipFile(CHECKPOINT) as archive:
        assert archive.testzip() is None
        assert {"data", "policy.pth", "policy.optimizer.pth", "_stable_baselines3_version"} <= set(archive.namelist())
        metadata = json.loads(archive.read("data"))
        assert metadata["policy_kwargs"]["net_arch"] == {"pi": [64, 64], "vf": [64, 64]}
