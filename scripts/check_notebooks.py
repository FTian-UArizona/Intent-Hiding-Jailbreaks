#!/usr/bin/env python3
"""Check notebook hygiene and Python syntax without executing research code.

Run from any directory with Python 3.10+: python scripts/check_notebooks.py.
No notebook, model, or GPU dependencies are needed. Git supplies the public file
list when available; the fallback excludes the repository's generated folders.
This is a static check, not a substitute for running experiments or a secret
scanner with access to repository history.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import re
import subprocess
import tokenize


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", ".work", ".ipynb_checkpoints", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "outputs", "node_modules",
}
EXCLUDED_PATHS = {"data/results", "data/queries"}
KEY_PATTERNS = (
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("API key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
)


def public_files() -> list[Path]:
    """Include tracked files even when an ignore rule would exclude them."""
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if Path(top).resolve() == ROOT:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=ROOT, check=True, capture_output=True,
            )
            names = {name.decode("utf-8", errors="surrogateescape")
                     for name in result.stdout.split(b"\0") if name}
            return sorted(ROOT / name for name in names if (ROOT / name).is_file())
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and not EXCLUDED_DIRS.intersection(path.relative_to(ROOT).parts)
        and not any(path.relative_to(ROOT).as_posix().startswith(prefix + "/")
                    for prefix in EXCLUDED_PATHS)
    )


def compile_cell(source: str, label: str) -> tuple[list[str], list[str]]:
    """Compile Python; explicitly disclose any IPython statements skipped."""
    errors: list[str] = []
    skipped: list[str] = []
    lines = source.splitlines(keepends=True)
    first = next((line.strip() for line in lines if line.strip()), "")
    if re.match(r"^%%[A-Za-z_]\w*\b", first):
        skipped.append(f"{label}: cell magic {first.split()[0]} (body not checked)")
        return errors, skipped

    # Token positions distinguish shell statements from text inside multiline
    # strings. Shell syntax can itself stop Python's tokenizer; tokens emitted
    # before that point still identify the surrounding strings and comments.
    protected: list[tuple[tuple[int, int], tuple[int, int]]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.STRING, tokenize.COMMENT):
                protected.append((token.start, token.end))
    except (tokenize.TokenError, IndentationError):
        pass
    for number, line in enumerate(lines, start=1):
        statement = line.lstrip()
        position = (number, len(line) - len(statement))
        inside_text = any(start <= position < end for start, end in protected)
        if not inside_text and (
            statement.startswith("!") or re.match(r"^%[A-Za-z_]\w*\b", statement)
        ):
            indentation = line[:len(line) - len(statement)]
            lines[number - 1] = indentation + "pass\n"
            skipped.append(f"{label}, line {number}: IPython shell/magic statement")
    try:
        compile("".join(lines), label, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    except SyntaxError as error:
        errors.append(f"{label}, line {error.lineno}: invalid Python syntax")
    return errors, skipped


def check_notebook(text: str, label: str) -> tuple[list[str], list[str], int]:
    errors: list[str] = []
    skipped: list[str] = []
    code_cells = 0
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return [f"{label}: invalid notebook JSON"], skipped, code_cells
    if not isinstance(notebook, dict):
        return [f"{label}: notebook must be a JSON object"], skipped, code_cells
    if notebook.get("nbformat") != 4:
        errors.append(f"{label}: expected notebook format 4")
    minor = notebook.get("nbformat_minor")
    if not isinstance(minor, int) or isinstance(minor, bool) or minor < 0:
        errors.append(f"{label}: missing or invalid notebook minor version")
    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) - {"kernelspec", "language_info"}:
        errors.append(f"{label}: keep only kernelspec/language_info notebook metadata")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return errors + [f"{label}: cells must be a list"], skipped, code_cells
    seen_ids: set[str] = set()
    for index, cell in enumerate(cells, start=1):
        cell_label = f"{label}, cell {index}"
        if not isinstance(cell, dict):
            errors.append(f"{cell_label}: cell must be an object")
            continue
        cell_id = cell.get("id")
        if cell_id is not None or (isinstance(minor, int) and minor >= 5):
            if not isinstance(cell_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id):
                errors.append(f"{cell_label}: missing or invalid cell ID")
            elif cell_id in seen_ids:
                errors.append(f"{cell_label}: duplicate cell ID")
            else:
                seen_ids.add(cell_id)
        metadata = cell.get("metadata")
        if not isinstance(metadata, dict) or set(metadata) - {"tags"}:
            errors.append(f"{cell_label}: keep only optional tags in cell metadata")
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            errors.append(f"{cell_label}: source must be text or a list of strings")
            continue
        kind = cell.get("cell_type")
        if kind not in {"code", "markdown", "raw"}:
            errors.append(f"{cell_label}: invalid cell type")
        if kind == "code":
            code_cells += 1
            if cell.get("outputs") != []:
                errors.append(f"{cell_label}: clear saved outputs")
            if "execution_count" not in cell or cell["execution_count"] is not None:
                errors.append(f"{cell_label}: clear execution count")
            cell_errors, cell_skipped = compile_cell(source, cell_label)
            errors.extend(cell_errors)
            skipped.extend(cell_skipped)
    return errors, skipped, code_cells


def main() -> int:
    errors: list[str] = []
    skipped: list[str] = []
    notebooks = code_cells = text_files = 0
    for path in public_files():
        label = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            errors.append(f"{label}: symlink requires manual review before publication")
            continue
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"{label}: exceeds the 10 MiB public-file limit")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "\0" in text:
            continue
        text_files += 1
        for name, pattern in KEY_PATTERNS:
            if pattern.search(text):
                errors.append(f"{label}: possible hardcoded {name}; value suppressed")
        if path.suffix == ".ipynb":
            notebooks += 1
            notebook_errors, notebook_skipped, count = check_notebook(text, label)
            errors.extend(notebook_errors)
            skipped.extend(notebook_skipped)
            code_cells += count
        elif path.suffix == ".py":
            try:
                compile(text, label, "exec")
            except SyntaxError as error:
                errors.append(f"{label}, line {error.lineno}: invalid Python syntax")
    if not notebooks:
        errors.append("No public notebooks found")
    for item in skipped:
        print(f"SKIP {item}")
    for item in errors:
        print(f"ERROR {item}")
    print(f"Checked {notebooks} notebooks, {code_cells} code cells, and {text_files} text files.")
    print("Static checks only: no cells, shell commands, dependencies, or models were executed.")
    print("Credential checks cover common literal patterns in current text files, not Git history.")
    print(f"{'FAIL' if errors else 'PASS'}: {len(errors)} error(s); {len(skipped)} disclosed skip(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
