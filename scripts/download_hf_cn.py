#!/usr/bin/env python3
"""Download Hugging Face models through a China-friendly endpoint.

Default: pinned, SHA-256-verified BasicVSR++ weight for Jasna.
Supports any public Hugging Face model file, or a whole repository snapshot.

Examples:
    /usr/bin/python3 scripts/download_hf_cn.py
    /usr/bin/python3 scripts/download_hf_cn.py --repo openai-community/gpt2 --snapshot
    /usr/bin/python3 scripts/download_hf_cn.py --repo org/model --file weights/model.safetensors
    /usr/bin/python3 scripts/download_hf_cn.py --official
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_REPO = "ladaapp/lada"
DEFAULT_FILE = "lada_mosaic_restoration_model_generic_v1.2.pth"
DEFAULT_REVISION = "3bfd69ffc21518bde80ba6b61696d51efd0a398b"
DEFAULT_SHA256 = "d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee"
MIRROR = "https://hf-mirror.com"
OFFICIAL = "https://huggingface.co"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="HF models: CN mirror (default), original endpoint, resume/cache, SHA-256"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo ID, e.g. org/model")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", help="One path inside the HF repository")
    source.add_argument("--snapshot", action="store_true",
                        help="Download an entire repo (rather than a single file)")
    parser.add_argument("--revision", help="Commit SHA, tag, or branch")
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"), default="model")
    parser.add_argument("--local-dir", type=Path,
                        help="Target directory; defaults to Jasna model_weights/")
    parser.add_argument("--sha256", help="Expected SHA-256 for one file")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", MIRROR),
                        help="HF endpoint; default https://hf-mirror.com")
    parser.add_argument("--official", action="store_true",
                        help="Use official https://huggingface.co instead")
    parser.add_argument("--use-token", action="store_true",
                        help="Allow HF_TOKEN authentication, official endpoint only")
    parser.add_argument("--disable-xet", action="store_true",
                        help="Use HTTP fallback if the mirror's Xet token endpoint fails")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for --snapshot (default: 4)")
    args = parser.parse_args(argv)

    if args.official:
        args.endpoint = OFFICIAL
    args.endpoint = args.endpoint.rstrip("/")
    u = urlsplit(args.endpoint)
    if u.scheme != "https" or not u.netloc or u.username or u.password or u.path:
        parser.error("--endpoint must be an HTTPS origin without a path or credentials")
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")
    if args.use_token and args.endpoint != OFFICIAL:
        parser.error("authentication tokens can only be used with --official")
    if args.snapshot and args.sha256:
        parser.error("--sha256 can only be checked for a single file")
    if not args.snapshot and args.file is None:
        if args.repo != DEFAULT_REPO or args.repo_type != "model":
            parser.error("for a different repo, specify --file or --snapshot")
        args.file = DEFAULT_FILE
    if args.file and (
        args.file.startswith("/") or
        any(p in (".", "..") for p in args.file.split("/")) or
        "\\" in args.file
    ):
        parser.error("--file must be a relative HF repository path")
    if args.revision is None:
        args.revision = (
            DEFAULT_REVISION
            if args.repo == DEFAULT_REPO and args.file == DEFAULT_FILE and not args.snapshot
            else "main"
        )
    if (
        args.sha256 is None and args.repo == DEFAULT_REPO
        and args.file == DEFAULT_FILE and args.repo_type == "model"
        and args.revision == DEFAULT_REVISION
    ):
        args.sha256 = DEFAULT_SHA256
    if args.sha256 and (
        len(args.sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.sha256)
    ):
        parser.error("--sha256 must be a 64-character hex digest")
    if args.local_dir is None:
        args.local_dir = (
            PROJECT_ROOT / "model_weights" / args.repo.split("/")[-1]
            if args.snapshot else PROJECT_ROOT / "model_weights"
        )
    return args


def main(argv=None):
    args = _parse_args(argv)
    args.local_dir.mkdir(parents=True, exist_ok=True)

    if not args.snapshot and args.sha256:
        target = args.local_dir / args.file
        if target.is_file() and sha256_file(target).lower() == args.sha256.lower():
            print("Already downloaded and verified:", target)
            return 0

    # Hugging Face reads these settings at IMPORT time.
    os.environ["HF_ENDPOINT"] = args.endpoint
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    # Never send ambient auth credentials to a third-party mirror.
    if args.endpoint != OFFICIAL:
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            os.environ.pop(key, None)
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("Missing dependency: install with /usr/bin/python3 -m pip install huggingface_hub",
              file=sys.stderr)
        return 2

    token = True if args.use_token else False
    print("Endpoint:", args.endpoint, flush=True)
    print("Repository:", args.repo, "revision:", args.revision, flush=True)
    try:
        if args.snapshot:
            result = snapshot_download(
                repo_id=args.repo, revision=args.revision, repo_type=args.repo_type,
                local_dir=str(args.local_dir), max_workers=args.workers, token=token,
            )
            print("Snapshot saved to:", result)
        else:
            result = hf_hub_download(
                repo_id=args.repo, filename=args.file, revision=args.revision,
                repo_type=args.repo_type, local_dir=str(args.local_dir), token=token,
            )
            result_path = Path(result)
            if args.sha256:
                actual = sha256_file(result_path)
                if actual.lower() != args.sha256.lower():
                    print("SHA-256 FAILED:", actual, "expected:", args.sha256, file=sys.stderr)
                    return 1
                print("SHA-256 verified:", actual)
            else:
                print("Warning: no expected SHA-256 provided; integrity not independently checked.")
            print("Model saved to:", result_path)
    except Exception as exc:
        print("Download failed:", str(exc), file=sys.stderr)
        print("Try --disable-xet (mirror token errors) or --official (official endpoint).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
