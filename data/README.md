# Data guide

## Included public data

| File or directory | Purpose |
| --- | --- |
| `task_alphabet_150_scored.csv` | The original 100 JailbreakBench targets and 50 Super-Natural Instructions auxiliary tasks with estimated harmful-intent probabilities |
| `processed/target_outcomes.csv` | Compact per-target, per-size counts for ASR reproduction; no generated prompts or responses |
| `processed/provenance.json` | ASR source-file hashes and aggregation provenance |
| `processed/flow_target_preservation.csv` | Flow-Judge means and valid/failed counts by condition and total bundle size |
| `processed/flow_provenance.json` | Canonical Flow-Judge checkpoint, coverage, and source hashes |
| `reference_tables/` | Three archived ASR tables checked by the regression tests |
| `provenance/` | Duplicate-upload inventory and flagged-generation provenance |
| `artifact_manifest.json` | Canonical raw query/result paths, hashes, and matching uploaded filenames |

The compact outcomes preserve the numerical information needed for the reported
ASR figures. They do not contain the text required to rejudge model responses.
The included alphabet retains its upstream task texts and experiment values.

## Import the original raw artifacts

Raw query/response tables are not included in this repository. To import them
from an extracted original `Intent` archive:

```bash
python scripts/prepare_data.py /path/to/Intent --group results
```

The script also accepts a directory containing the separately supplied files,
including the duplicate upload suffixes listed in the manifest:

```bash
python scripts/prepare_data.py /path/to/uploaded_files --group results
python scripts/prepare_data.py /path/to/uploaded_files --generator qwen3_32b
```

Use `--group queries` for query tables, `--generator qwen3_14b` for the secondary
generator, and `--dry-run` to verify without copying. The original archive is
needed to import the optional Gemma generator's query table if it is absent
from the uploaded-file directory.

Import verifies every selected size and SHA-256 hash, accepts only recorded
filename aliases, and refuses to overwrite different existing data. Files are
placed in Git-ignored `queries/` and `results/` directories. The manifest selects
the complete 10,100-row Qwen3-1.7B/14B workbook; the older figure-directory copy
has one missing direct baseline. See `../docs/REPRODUCIBILITY.md` for the exact
comparison supporting that selection.

To rebuild compact outcomes and figures from canonical raw results
(XLSX input also requires `openpyxl`, included in `requirements.txt`):

```bash
python scripts/plot_results.py --from-results data/results --output outputs/recomputed
```

This also saves newly aggregated counts and source hashes under the output
directory. For the raw Flow-Judge archive or checkpoint, follow
`../docs/FLOW_RESULTS.md`.

## Excluded data

Large response tables, raw judgment checkpoints, model/server logs, duplicate
exports, and exploratory pilot outputs are excluded from Git. The separate
726-row GPT-OSS pilot is not combined with the paper's 100-target conditions.
The refusal manifest is retained only as compact provenance; it does not imply
that its flagged candidates were replaced.

Upstream datasets retain their original terms. No new license is assigned to
third-party data by this repository.
