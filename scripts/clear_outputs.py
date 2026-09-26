#!/usr/bin/env python3
"""Remove saved notebook outputs and private execution metadata before committing."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    changed = 0
    for path in notebooks:
        original = path.read_text(encoding="utf-8")
        notebook = json.loads(original)
        notebook["metadata"] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        notebook["nbformat"] = 4
        notebook["nbformat_minor"] = 5
        for index, cell in enumerate(notebook["cells"]):
            cell["metadata"] = {}
            cell.setdefault(
                "id", hashlib.sha256(f"{path.relative_to(ROOT)}:{index}".encode()).hexdigest()[:12]
            )
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        updated = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    print(f"Cleared outputs and metadata in {changed} of {len(notebooks)} notebooks.")


if __name__ == "__main__":
    main()
