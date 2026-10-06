"""Download and extract the Brain Stroke MRI Images dataset into dataset/raw/.

Usage (project root, venv active):
    python scripts/download_dataset.py            # skip if already present
    python scripts/download_dataset.py --force    # re-download

Source: https://www.kaggle.com/datasets/mitangshu11/brain-stroke-mri-images
The public Kaggle download endpoint is used; no API token is required for
this public dataset at the time of writing. If the download fails, download
the archive manually from the URL above and place it at
``dataset/raw/brain-stroke-mri-images.zip``, then re-run this script.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.image_audit import file_sha256  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("download_dataset")


def download(url: str, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".part")
    logger.info("Downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "brain-stroke-detection/phase2"})
    with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)
    tmp.replace(destination)
    logger.info("Saved %s (%.1f MB)", destination, destination.stat().st_size / 1e6)


def safe_extract(archive: Path, target: Path) -> None:
    """Extract a zip, refusing members that would escape ``target`` (zip-slip)."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if target not in (target / member.filename).resolve().parents and (target / member.filename).resolve() != target:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = parser.parse_args()

    config = load_config()
    dataset = config["dataset"]
    raw_dir = resolve_path(config["paths"]["raw_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / dataset["archive_name"]
    extract_dir = raw_dir / dataset["extract_dir"]

    if args.force or not archive.is_file():
        try:
            download(dataset["download_url"], archive)
        except Exception as exc:  # noqa: BLE001
            logger.error("Download failed: %s", exc)
            logger.error("Download manually from %s and save it as %s", dataset["source_url"], archive)
            return 1
    else:
        logger.info("Archive already present: %s", archive)

    digest = file_sha256(archive)
    if digest != dataset["archive_sha256"]:
        logger.warning(
            "SHA-256 mismatch — upstream dataset may have changed.\n  expected %s\n  got      %s\n"
            "Statistics in README.md were computed on the expected version.",
            dataset["archive_sha256"], digest,
        )
    else:
        logger.info("SHA-256 verified")

    if args.force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if not extract_dir.exists():
        safe_extract(archive, extract_dir)
        logger.info("Extracted to %s", extract_dir)
    else:
        logger.info("Already extracted: %s", extract_dir)

    source = extract_dir / dataset["source_subdir"]
    if not source.is_dir():
        logger.error("Expected source folder not found after extraction: %s", source)
        return 1
    logger.info("Source folder ready: %s", source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
