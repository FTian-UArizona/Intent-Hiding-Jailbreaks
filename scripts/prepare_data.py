#!/usr/bin/env python3
"""Restore canonical experiment artifacts from an extracted original Intent folder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes its root: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted Intent directory or folder of supplied files")
    parser.add_argument(
        "--group", choices=("all", "queries", "results"), default="all"
    )
    parser.add_argument(
        "--generator", choices=("all", "qwen3_32b", "qwen3_14b", "gemma3_27b_pt"),
        default="all", help="Import only one generator's artifacts"
    )
    parser.add_argument("--dry-run", action="store_true", help="Verify without copying")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Source directory does not exist: {source}")
    manifest = json.loads((ROOT / "data/artifact_manifest.json").read_text())
    entries = [
        item for item in manifest["files"]
        if (args.group == "all" or item["destination"].startswith(f"data/{args.group}/"))
        and (args.generator == "all" or f"/{args.generator}/" in item["destination"])
    ]

    # Validate the complete selection before copying; never silently overwrite results.
    pending = []
    for item in entries:
        candidates = [contained_path(source, name) for name in
                      [item["source"], *item.get("uploaded_names", [])]]
        existing = [path for path in candidates if path.is_file()]
        original = next(
            (path for path in existing if path.stat().st_size == item["size_bytes"]
             and sha256(path) == item["sha256"]), None
        )
        target = contained_path(ROOT, item["destination"])
        if original is None:
            problem = "No matching archived content" if existing else "Missing source"
            parser.error(f"{problem}: {item['source']}. Use --generator to import a subset.")
        if target.exists():
            if not target.is_file() or sha256(target) != item["sha256"]:
                parser.error(f"Refusing to overwrite a different file: {item['destination']}")
        else:
            pending.append((original, target))
    for original, target in pending:
        print(("Would copy: " if args.dry_run else "Copying: ") + str(target.relative_to(ROOT)))
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    print(f"Verified {len(entries)} files; {len(pending)} {'to copy' if args.dry_run else 'copied'}.")


if __name__ == "__main__":
    main()
