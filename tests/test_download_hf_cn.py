"""Argument and integrity tests for the HF China mirror downloader.

No network, Hugging Face credentials or model weight downloads are required.
"""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_hf_cn.py"
spec = importlib.util.spec_from_file_location("download_hf_cn", SCRIPT)
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def test_basicvsrpp_default_pinned_and_verified():
    args = dl._parse_args([])
    assert args.repo == "ladaapp/lada"
    assert args.file == "lada_mosaic_restoration_model_generic_v1.2.pth"
    assert args.revision == dl.DEFAULT_REVISION
    assert args.sha256 == dl.DEFAULT_SHA256
    assert args.endpoint == "https://hf-mirror.com"


def test_official_custom_model_and_revision(tmp_path):
    args = dl._parse_args([
        "--repo", "org/some-model", "--file", "weights/model.safetensors",
        "--revision", "abc123", "--official", "--local-dir", str(tmp_path),
    ])
    assert args.endpoint == "https://huggingface.co"
    assert args.sha256 is None
    assert args.revision == "abc123"
    assert args.file == "weights/model.safetensors"


def test_snapshot_defaults(tmp_path):
    args = dl._parse_args(["--repo", "org/model", "--snapshot", "--workers", "8"])
    assert args.snapshot
    assert args.local_dir.name == "model"
    assert args.revision == "main"
    assert args.workers == 8


@pytest.mark.parametrize("args", [
    ["--repo", "other/repo"],
    ["--file", "../secret"],
    ["--file", "/tmp/secret"],
    ["--file", r"foo\bar"],
    ["--endpoint", "http://hf-mirror.com"],
    ["--endpoint", "https://user:password@hf-mirror.com"],
    ["--use-token"],
    ["--workers", "0"],
    ["--snapshot", "--sha256", "0" * 64],
    ["--sha256", "invalid"],
])
def test_rejects_invalid_inputs(args):
    with pytest.raises(SystemExit):
        dl._parse_args(args)


def test_sha256_file(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"sample-weight")
    assert dl.sha256_file(path) == hashlib.sha256(b"sample-weight").hexdigest()


def test_cached_valid_file_skips_hf_hub(tmp_path):
    path = tmp_path / dl.DEFAULT_FILE
    path.write_bytes(b"test")
    digest = dl.sha256_file(path)
    assert dl.main(["--local-dir", str(tmp_path), "--sha256", digest]) == 0
