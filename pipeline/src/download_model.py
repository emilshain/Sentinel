"""
Fetch the backdoored reference model.

The checkpoint is 255 MB, over GitHub's 100 MB per-file limit, so it ships as a
Release asset instead of living in the repository. The repo is public, so no
token or authentication is involved.

The SHA-256 below is verified before anything is unpacked. That is not ceremony:
Sentinel's entire thesis is that you cannot trust an artifact just because it
arrived from the expected place, and a supply-chain auditing tool that unzips an
unverified 246 MB download would be making exactly the mistake it exists to catch.

    python src/download_model.py

Nothing here is required to see Sentinel work - the repository ships a real
recorded run (reports/golden_run.json) and the demo falls back to it. The model
is only needed to execute a *live* scan.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile


PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSET_URL = (
    "https://github.com/emilshain/Sentinel/releases/download/v1.0/backdoor_model.zip"
)
ASSET_SHA256 = "32cb0c41edf37ffd9036a4b2c4ef97024b17ee4180ed42bf21277cb5bd81b5bf"
ASSET_BYTES = 246989001

CHECKPOINT_DIR = os.path.join(PIPELINE_ROOT, "model_checkpoints", "backdoor_model")
WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")


def weights_present():
    return any(os.path.isfile(os.path.join(CHECKPOINT_DIR, n)) for n in WEIGHT_NAMES)


def _progress(done, total):
    if not sys.stdout.isatty() or not total:
        return
    pct = done * 100 // total
    bar = "#" * (pct // 3)
    sys.stdout.write(f"\r  [{bar:<33}] {pct:3d}%  {done // 1048576}/{total // 1048576} MB")
    sys.stdout.flush()


def download(url, destination):
    print(f"[model] downloading {url}")
    digest = hashlib.sha256()
    downloaded = 0

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or ASSET_BYTES)
        with open(destination, "wb") as out:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                _progress(downloaded, total)
    if sys.stdout.isatty():
        print()
    return digest.hexdigest(), downloaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=ASSET_URL)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if weights exist"
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="skip SHA-256 verification (not recommended)",
    )
    args = parser.parse_args()

    if weights_present() and not args.force:
        print(f"[model] weights already present in {CHECKPOINT_DIR} - nothing to do.")
        print("[model] re-download with --force")
        return 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "backdoor_model.zip")
        try:
            actual, size = download(args.url, archive)
        except Exception as exc:
            print(f"\n[model] download failed: {type(exc).__name__}: {exc}")
            print(f"[model] the asset should be at:\n         {args.url}")
            print("[model] if it 404s the release may not be published yet; the demo "
                  "still works without weights via reports/golden_run.json")
            return 1

        print(f"[model] downloaded {size} bytes")

        if not args.skip_checksum:
            if actual != ASSET_SHA256:
                print("[model] CHECKSUM MISMATCH - refusing to unpack.")
                print(f"         expected {ASSET_SHA256}")
                print(f"         actual   {actual}")
                print("[model] the download was corrupted or the asset was replaced.")
                return 2
            print(f"[model] sha256 verified: {actual}")

        print(f"[model] unpacking into {CHECKPOINT_DIR}")
        with zipfile.ZipFile(archive) as zf:
            # The archive contains a top-level backdoor_model/ directory; strip it
            # so files land directly in model_checkpoints/backdoor_model/.
            for member in zf.infolist():
                if member.is_dir():
                    continue
                relative = member.filename.split("/", 1)[-1]
                if not relative:
                    continue
                target = os.path.join(CHECKPOINT_DIR, relative)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    if not weights_present():
        print("[model] unpacked, but no weights file found - archive layout unexpected.")
        return 3

    print(f"[model] done. Live scans are now possible.")
    print(f"[model] verify with: python src/demo_runner.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
