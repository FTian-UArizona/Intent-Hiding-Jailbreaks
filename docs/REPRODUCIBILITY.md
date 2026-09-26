# Reproduction notes

## Preserved experiment settings

The experiment notebooks preserve the original generation and inference prompts, HarmBench
templates, Flow-Judge rubric and rubric hash, model identifiers, decoding
settings, master seeds, auxiliary sampling, and water-filling functions.
The repository does not claim a new GPU reproduction of the paper.

The 32B generator creates 10,000 composed candidates: 100 targets, five
auxiliary counts, and 20 candidates per target/count. The matched 14B and Gemma
generators reuse those bundles. Inference adds the 100 direct requests.
Generation reads and writes CSV as its primary interchange format; matched
generators also retain the original XLSX export.

The alphabet builder retains its existing `APPROVE_DICTIONARY_FOR_SCORING=True`
default. Review that setting and the generated task collection if rebuilding
the alphabet. The included alphabet is the archived collection used in the
CPU validation.

## Result-source selection

`data/artifact_manifest.json` records every imported source, destination, size,
and SHA-256 hash. Result selection follows the current figure notebook, with
one verified recovery:

- The figure-directory Qwen3-1.7B/14B workbook contains 10,099 rows and omits
  `direct_jbb_099`.
- The corresponding workbook under `14B Inference/Qwen1.7B/` has 10,100 rows.
  All 38 fields in all 10,099 shared rows match exactly, including response
  text, response hashes, labels, targets, and bundle metadata.
- The import manifest selects that complete workbook. This supplies the
  original missing response without fabricating a row or changing shared data.
  Figure D is unchanged because the original figure procedure replaces every
  14B direct outcome with the corresponding 32B direct outcome.

The strict Flow-Judge input loader validates all 80,800 rows across its eight
configured conditions with the complete workbook. This validation does not
establish that every GPU judgment has been completed.

## Runtime limitations retained from the source

| Area | Limitation |
| --- | --- |
| Environment | vLLM, several supporting packages, model revisions, and dataset revisions are not fully pinned. Save installed package versions and resolved revisions for new runs. |
| Resumed 32B generation | The pending-job loop restarts batch indexing, so resumed stochastic outputs need not match an uninterrupted run bit for bit. |
| Legacy inference summaries | Some metrics summarize only rows present after generation; separately logged failures can be omitted. Use the full experimental roster for paper-level comparisons. |
| Mistral | The original first judge pass uses the generic `behavior` field. Run the corrected rejudge notebook against full `target_text`. Only the corrected step exports the canonical result for analysis. |
| Mistral/Gemma recovery | Their recovery notebooks require the complete expected result count; Mistral additionally requires nonblank responses. |
| Flow-Judge | Its original loader requires 10,100 rows per condition and nonmissing response text. Empty or failed generations may require explicit failure handling before evaluating a new run. |
| Optional API judge | The original resume logic trusts an existing output file and does not compare input hashes. Error/block markers count as populated and are not automatically retried. Use a fresh output path for a different input. |

The experiment notebooks include both per-candidate and target-level metrics.
For the paper, `scripts/plot_results.py` computes target-level cumulative ASR
and incremental gains with exactly 100 targets. The compact input validator
requires all ten configured conditions, all six total bundle sizes, one direct
candidate per target, and twenty candidates per target at each composed size.
Raw-result extraction rejects malformed labels and duplicate candidate IDs;
missing labels continue to count as failed attacks.

The published compact data is built from complete inputs, including the
verified recovered 14B workbook. Figure D then replaces the 14B direct outcomes
with the 32B outcomes before computing the cumulative target unions. This
replacement is tested across the full curve, not just at the first point.

The ASR figure loader does not independently judge response text. It relies on
the stored HarmBench labels. Ensure failed or blank responses have failure
labels in any new result files. The experiment notebooks preserve existing refusal and
error handling; they do not retry until success or change recorded outcomes.

## Data and naming

Historical filenames containing `k1_to_k5` refer to auxiliary counts, not total
bundle sizes. They are retained for compatibility. In plots, total sizes are
`K=1,...,6`, including the direct baseline.

Historical column names such as `tfs`, `flow_judge_scores`, and
`task_alphabet_150_scored.csv` remain stable for checkpoint compatibility.
These internal names do not change the paper's terminology: prior–posterior
matching, ASR, and target-preservation level.

The raw query and response artifacts are excluded from Git tracking. `prepare_data.py` restores them from the original archive; new
experiments generate them locally. The small reference tables in
`data/reference_tables/` record the validated ASR exports.

## Compact public outcomes

`data/processed/target_outcomes.csv` stores the candidate count and HarmBench
success count for each generator, downstream model, target, and total bundle
size. It contains no generated query or response text. The figure command uses
these counts to recover per-target success and the original cumulative ASR.
This supports numerical reproduction of the reported figures; rejudging
responses requires the original raw data.

`data/processed/flow_target_preservation.csv` reports counts and normalized
levels from the selected Flow-Judge run. See `FLOW_RESULTS.md` for exact
checkpoint selection and coverage. Historical filenames and field names are
retained in provenance, while displayed metrics use target-preservation level.

## Supplemental-upload audit

All 28 ordinary notebook/CSV/XLSX uploads matched existing archive files exactly.
Repeated filename suffixes identify duplicate uploads, not new experimental
runs. The complete 14B Qwen3-1.7B workbook is among those uploads.

The separate `scored_results.csv` contains a 726-row, 10-target GPT-OSS-120B pilot
with three exploratory profiles. It is excluded from the main paper results.
The refusal-regeneration manifest lists 13 flagged candidates, but its recorded
old queries still match both `generated_query` and `prompt` in the supplied 32B
candidate table. It therefore provides no evidence of completed replacements.
`data/provenance/` records candidate identifiers, query hashes, source hashes,
and selection decisions without publishing those raw queries.
