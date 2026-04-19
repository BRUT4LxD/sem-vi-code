#!/usr/bin/env python3
"""
Deduplicate images in place (recursive).

For each file content hash, keeps the first path (lexicographically) and removes
all other duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def list_images(root: Path, extensions: tuple[str, ...]) -> List[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate images in place from a folder recursively."
    )
    parser.add_argument("--root", required=True, help="Root folder to scan recursively")
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".webp"],
        help="Image extensions to include",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete files, only print summary",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    extensions = tuple(x.lower() for x in args.ext)
    files = list_images(root, extensions)
    print(f"Root: {root}")
    print(f"Images found: {len(files)}")

    by_hash: Dict[str, List[Path]] = {}
    for path in tqdm(files, desc="Hashing", unit="img"):
        h = sha256_file(path)
        by_hash.setdefault(h, []).append(path)

    duplicates_to_remove: List[Path] = []
    for paths in by_hash.values():
        if len(paths) > 1:
            duplicates_to_remove.extend(paths[1:])

    removed = 0
    for path in tqdm(duplicates_to_remove, desc="Removing duplicates", unit="file"):
        if args.dry_run:
            print(f"Would delete: {path}")
            continue
        try:
            path.unlink()
            removed += 1
        except Exception as e:  # noqa: BLE001
            print(f"Failed to delete {path}: {e}")

    print(f"Duplicate files found: {len(duplicates_to_remove)}")
    if args.dry_run:
        print("Dry-run mode: no files deleted.")
    else:
        print(f"Deleted: {removed}")


if __name__ == "__main__":
    main()
