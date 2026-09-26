# Response-level target preservation

[`flow_target_preservation.csv`](../data/processed/flow_target_preservation.csv)
contains 48 condition–bundle-size groups from **80,800 valid saved judgments**.
The archived checkpoint in `flow_judge_tfs_v01-20260925T094525Z-1-001.zip`
is the canonical source. No model inference was rerun to prepare this release.

The evaluation covers five downstream checkpoints with the Qwen3-32B query
generator (Qwen3-1.7B, Qwen3-4B, Qwen3-4B-Instruct-2507, Qwen3-8B, and
Mistral-7B-Instruct-v0.3), and the first three Qwen checkpoints with the
Qwen3-14B query generator. **Gemma has no Flow-Judge results in this release.**
Each condition has 100 direct requests and 2,000 candidates at each total
bundle size from 2 to 6, giving 10,100 responses per condition.

## Reading the table

| Column | Meaning |
| --- | --- |
| `condition` | Stable identifier for the generator/downstream combination. |
| `generator`, `model` | Query-generator size and downstream checkpoint. |
| `K` | Total bundle size: one designated target plus `K - 1` auxiliary tasks. `K = 1` is a direct request. |
| `n_total` | Number of source responses in this condition and bundle size. |
| `n_valid` | Judgments with a valid 1–5 rubric value, the expected rubric, and the exact source input hash. |
| `n_failed` | Source rows without a valid matching judgment; zero throughout this release. This is not the number of failed attempts in the append-only checkpoint. |
| `mean_level` | Mean target-preservation level, normalized to `[0, 1]`. |
| `n_repaired` | Valid judgments obtained in the saved bounded-feedback retry pass. |
| `n_response_hash_mismatch` | Source rows whose stored response SHA256 differs from the actual response text. |

The judge is `flowaicom/Flow-Judge-v0.1`. It assesses the response against the
original designated target; auxiliary-task completion alone earns no credit.
The five rubric levels are normalized as `(level_1to5 - 1) / 4`. The mean uses
all valid responses in the group, including refusals and low-preservation
responses; it is not conditioned on HarmBench success. The groups are for each
bundle size individually, not cumulative best-of results.

If an input checkpoint is incomplete, the script reports unresolved rows in
`n_failed` and computes `mean_level` over `n_valid` only. It never replaces a
missing judgment with zero. A group with no valid judgments has an empty mean.
All published groups are complete, so `n_valid = n_total` here.

The direct-request rows retain the Flow-Judge judgments from each condition's
own source file. No Qwen3-32B direct-request values are substituted into the
Qwen3-14B Flow-Judge results.

## Why this checkpoint was selected

The standalone `flow_judge_scores(3).jsonl` contains 80,584 valid judgments and
216 unresolved rows. The archived `flow_judge_scores.jsonl` contains those
same 80,584 judgments with identical values and input hashes, plus 216 valid
retry judgments. It covers all 80,800 source rows. The selection therefore
depends on verified coverage and input identity, not the upload suffix or
filename date.

The append-only complete checkpoint retains 730 historical error records.
These are failed attempts that were subsequently resolved, not 730 missing
results. The 216 retries use the saved
`same_prompt_bounded_feedback_regex_v1` protocol with the same rubric and
source inputs. Every accepted retry contains a complete feedback-and-value
structure. Of these, 215 records report a `stop` finish reason and one reports
`length`; the latter is retained because its complete output passes the same
structural validation and contains a single valid rubric value.

The saved rubric SHA256 is
`8e3d3a7fa5ccd1262ec50c5f13eebae0a279c385b6bd040f06df00654068542e`.
It matches the rubric in
[`flow_judge_target_preservation.ipynb`](../notebooks/evaluation/flow_judge_target_preservation.ipynb).
Full source, export, checkpoint, and summary hashes are recorded in
[`flow_provenance.json`](../data/processed/flow_provenance.json).

## Verification and limitations

The offline audit resolves all eight inference files by their manifest
SHA256, reconstructs each input hash from the actual original target and
response text, and accepts only matching checkpoint records. Every one of the
80,800 accepted rubric values agrees with the exported per-response table.
Regrouping those records reproduces the supplied direct-request and composed
summary means.

Nine Mistral source rows have a historical stored response-hash mismatch.
These rows remain flagged in the summary and provenance. Their Flow-Judge
input hashes match the actual text in the exact manifested source file, so
they are included in these target-preservation means. This check does not
establish why the older response-hash fields differ or independently validate
their HarmBench labels.

The large raw checkpoint, judge feedback, server log, and duplicate archives
are omitted from the public repository. Their hashes identify the originals
needed for a full offline reconstruction; the compact table is sufficient to
inspect and plot the reported means. The repository alone does not contain
the raw experiment inputs required for the audit command below.

## Rebuild from the original artifacts

Extract the archived checkpoint and `flow_judge_tfs_results.zip` locally.
Point `--sources` at a folder containing the original eight inference
CSV/XLSX files; filenames may differ because selection uses file hashes.
From the repository root:

```bash
python scripts/summarize_flow.py --help

python scripts/summarize_flow.py \
  --rows /path/to/flow_judge_tfs_rows.csv \
  --manifest /path/to/flow_judge_tfs_manifest.json \
  --checkpoint /path/to/flow_judge_scores.jsonl \
  --sources /path/to/original-inference-files
```

Optionally pass `--compare-checkpoint /path/to/older-checkpoint.jsonl` to record
coverage and value agreement with another saved checkpoint. `openpyxl` is
required to stream the XLSX sources. This command uses no GPU, API, or network.
