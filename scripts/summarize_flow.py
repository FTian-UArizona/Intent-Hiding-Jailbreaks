#!/usr/bin/env python3
"""Audit saved Flow-Judge outputs and summarize target-preservation levels offline.

The source files are selected by the SHA256 values in the experiment manifest,
not by their filenames. No model inference or network access is performed.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SCORE = re.compile(r"<score>\s*([1-5])\s*</score>", re.IGNORECASE)
REPAIR = re.compile(r"\s*<feedback>[^<]{0,400}</feedback>\s*<score>[1-5]</score>\s*")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    if path.suffix.lower() == ".xlsx":
        import openpyxl

        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook.active.iter_rows(values_only=True)
            columns = next(rows)
            for values in rows:
                # Match pandas.read_excel's numeric-cell conversion used by
                # the experiment notebook (e.g. an Excel numeric 3.0 -> 3).
                values = [int(value) if type(value) is float and value.is_integer()
                          else value for value in values]
                yield dict(zip(columns, values))
        finally:
            workbook.close()
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


def integer(value):
    number = float(value)
    if not number.is_integer():
        raise ValueError("Expected an integer-valued field")
    return int(number)


def load_checkpoint(path, rubric_hash, expected):
    """Keep the last valid judgment for the exact current input of each row."""
    accepted, counts = {}, Counter()
    protocols, finish_reasons = Counter(), Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counts["n_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["n_malformed_lines"] += 1
                continue
            if not isinstance(record, dict):
                counts["n_malformed_records"] += 1
                continue
            if record.get("status") != "ok":
                counts["n_error_or_other_records"] += 1
                continue
            key = record.get("job_key")
            if record.get("rubric_hash") != rubric_hash:
                counts["n_wrong_rubric_records"] += 1
                continue
            if key not in expected or record.get("input_hash") != expected[key]:
                counts["n_unmatched_input_records"] += 1
                continue
            score = record.get("score_1to5")
            matches = SCORE.findall(record.get("raw_output") or "")
            if (type(score) not in (int, float) or score not in (1, 2, 3, 4, 5)
                    or len(matches) != 1 or int(matches[0]) != score):
                counts["n_invalid_judgment_records"] += 1
                continue
            protocol = record.get("retry_protocol")
            if protocol:
                if (protocol != "same_prompt_bounded_feedback_regex_v1"
                        or not REPAIR.fullmatch(record.get("raw_output") or "")):
                    counts["n_invalid_repair_records"] += 1
                    continue
                protocols[protocol] += 1
                finish_reasons[str(record.get("finish_reason"))] += 1
            accepted[key] = {"score": int(score), "repaired": bool(protocol)}
            counts["n_valid_records"] += 1
    report = {"file": path.name, "sha256": sha256(path), **dict(counts),
              "n_expected": len(expected), "n_valid": len(accepted),
              "n_failed": len(expected) - len(accepted),
              "repair_protocols": dict(protocols),
              "repair_finish_reasons": dict(finish_reasons)}
    return accepted, report


def summarize(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rubric_hash = hashlib.sha256("\n".join(manifest[field] for field in (
        "judge_model", "rubric_version", "criteria", "rubric")).encode()).hexdigest()
    if rubric_hash != manifest["rubric_sha256"]:
        raise ValueError("Manifest rubric hash is inconsistent")
    if manifest.get("normalization") != "(score_1to5 - 1) / 4":
        raise ValueError("Unexpected normalization")
    if manifest.get("score_scope") != "all":
        raise ValueError("Expected the all-response evaluation scope")

    # Resolve originals by content. Duplicate uploads with different names agree.
    wanted_hashes = {spec["sha256"] for spec in manifest["source_files"]}
    sources = {}
    for path in sorted(args.sources.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".csv", ".xlsx"):
            digest = sha256(path)
            if digest in wanted_hashes:
                sources.setdefault(digest, path)
    if sources.keys() != wanted_hashes:
        raise ValueError("Source directory is missing files identified by manifest SHA256")

    expected, metadata, source_report = {}, {}, []
    for spec in manifest["source_files"]:
        condition, counts, mismatches = spec["condition"], Counter(), 0
        for row in read_rows(sources[spec["sha256"]]):
            key = condition + "|" + str(row["row_id"])
            if key in expected:
                raise ValueError("Duplicate source job key")
            target, response = row["target_text"], row["response"]
            if target is None or response is None or not str(target).strip():
                raise ValueError("Missing source target or response")
            expected[key] = hashlib.sha256(
                (rubric_hash + "\n" + str(target) + "\n" + str(response)).encode()
            ).hexdigest()
            k_aux = integer(row["k_aux"])
            mismatch = hashlib.sha256(str(response).encode()).hexdigest() != row["response_sha256"]
            metadata[key] = {
                "condition": condition, "k_aux": k_aux,
                "target_id": str(row["target_id"]),
                "candidate_id": str(row["candidate_id"]),
                "response_hash_mismatch": mismatch,
            }
            counts[k_aux] += 1
            mismatches += mismatch
        if counts != {0: 100, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000}:
            raise ValueError("Unexpected source bundle-size coverage")
        if mismatches != spec["n_response_hash_mismatch"]:
            raise ValueError(f"{condition}: response hash mismatches {mismatches} disagree with manifest {spec['n_response_hash_mismatch']}")
        source_report.append({**spec, "n_rows": sum(counts.values()),
                              "matched_file": sources[spec["sha256"]].name})

    scores, checkpoint_report = load_checkpoint(args.checkpoint, rubric_hash, expected)
    comparisons = []
    for path in args.compare_checkpoint:
        other, report = load_checkpoint(path, rubric_hash, expected)
        shared = other.keys() & scores.keys()
        report.update(n_shared=len(shared),
                      n_shared_score_disagreements=sum(other[k]["score"] != scores[k]["score"] for k in shared),
                      n_only_selected=len(scores.keys() - other.keys()),
                      n_only_comparison=len(other.keys() - scores.keys()))
        comparisons.append(report)

    groups, seen = defaultdict(list), set()
    for row in read_rows(args.rows):
        key = row["condition"] + "|" + str(row["row_id"])
        if key in seen or key not in metadata:
            raise ValueError("Duplicate or unknown exported row key")
        seen.add(key)
        meta = metadata[key]
        if (integer(row["k_aux"]) != meta["k_aux"]
                or str(row["target_id"]) != meta["target_id"]
                or str(row["candidate_id"]) != meta["candidate_id"]
                or (str(row["response_sha_match"]).lower() == "false") != meta["response_hash_mismatch"]):
            raise ValueError("Exported row metadata disagrees with source")
        judgment = scores.get(key)
        if judgment and (float(row["score_1to5"]) != judgment["score"]
                         or float(row["tfs_0to1"]) != (judgment["score"] - 1) / 4):
            raise ValueError("Exported judgment disagrees with matched checkpoint")
        group = (row["condition"], row["generator_display"], row["model_display"], meta["k_aux"] + 1)
        groups[group].append((judgment, meta["response_hash_mismatch"]))
    if seen != expected.keys() or len(expected) != manifest["n_source_rows"]:
        raise ValueError("Export does not cover exactly the manifested source rows")

    summary = []
    for (condition, generator, model, k), items in sorted(groups.items()):
        valid = [judgment for judgment, _ in items if judgment is not None]
        levels = [(judgment["score"] - 1) / 4 for judgment in valid]
        summary.append({"condition": condition, "generator": generator, "model": model,
                        "K": k, "n_total": len(items), "n_valid": len(valid),
                        "n_failed": len(items) - len(valid),
                        "mean_level": sum(levels) / len(levels) if levels else "",
                        "n_repaired": sum(judgment["repaired"] for judgment in valid),
                        "n_response_hash_mismatch": sum(mismatch for _, mismatch in items)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    provenance = {
        "judge_model": manifest["judge_model"], "rubric_version": manifest["rubric_version"],
        "rubric_sha256": rubric_hash, "criteria": manifest["criteria"], "rubric": manifest["rubric"],
        "normalization": manifest["normalization"], "K_definition": "k_aux + 1",
        "denominator": "All valid, exact-input-matched judgments in the condition and bundle size; unresolved rows are excluded, never assigned zero.",
        "checkpoint_selection": "Last valid record for each job_key with matching rubric and source-derived input hash; later errors do not erase a valid judgment.",
        "all_rows_complete": len(scores) == len(expected),
        "n_total": len(expected), "n_valid": len(scores), "n_failed": len(expected) - len(scores),
        "n_repaired": sum(record["repaired"] for record in scores.values()),
        "n_response_hash_mismatch": sum(meta["response_hash_mismatch"] for meta in metadata.values()),
        "checkpoint": checkpoint_report, "compared_checkpoints": comparisons,
        "export_rows": {"file": args.rows.name, "sha256": sha256(args.rows)},
        "export_manifest": {"file": args.manifest.name, "sha256": sha256(args.manifest)},
        "source_files": source_report,
        "summary": {"file": args.output.name, "sha256": sha256(args.output), "n_groups": len(summary)},
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(summary)} groups: {len(scores):,}/{len(expected):,} valid judgments.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="Exported flow_judge_tfs_rows.csv")
    parser.add_argument("--manifest", type=Path, required=True, help="Exported flow_judge_tfs_manifest.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Selected saved JSONL checkpoint")
    parser.add_argument("--sources", type=Path, required=True, help="Folder containing original inference CSV/XLSX files")
    parser.add_argument("--compare-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("data/processed/flow_target_preservation.csv"))
    parser.add_argument("--provenance", type=Path, default=Path("data/processed/flow_provenance.json"))
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
