# Running the experiments

The default CPU figure command uses compact published outcomes. The steps below
run or resume the underlying model experiments. Use Python 3.11 and start
Jupyter from the repository root:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

## Notebook order

1. Use `data/task_alphabet_150_scored.csv`, or rebuild it with
   `notebooks/alphabet/build_task_alphabet.ipynb`.
   `notebooks/alphabet/waterfilling.ipynb` reproduces the fractional matching
   analysis on CPU.
2. Run `notebooks/generation/qwen3_32b.ipynb` to generate the primary candidate
   pool. For the matched generator comparison, then run
   `notebooks/generation/qwen3_14b.ipynb`, which reuses its stored bundles.
3. Run the desired downstream notebook under `notebooks/inference/qwen3_32b/`
   or `notebooks/inference/qwen3_14b/`.
4. For Mistral, run `mistral_7b.ipynb` followed by
   **`mistral_7b_rejudge.ipynb`**. The original first judge pass uses a generic
   behavior field; the corrected pass uses full `target_text`. Only that pass
   exports the canonical Mistral result for analysis.
5. Run `notebooks/evaluation/flow_judge_target_preservation.ipynb` after all
   eight configured inputs pass validation. This performs local judging on GPU.
6. Rebuild compact outcomes and figures with
   `python scripts/plot_results.py --from-results data/results --output outputs/recomputed`.
   Use `scripts/summarize_flow.py` to rebuild the target-preservation summaries
   from a saved checkpoint; see `FLOW_RESULTS.md` for that command.

The optional Gemma query generator and paid API judge comparison are under
`notebooks/optional/`. They are not prerequisites for the default figure command.

## Paths and outputs

Notebook setup cells locate the repository from the working directory. If
launching elsewhere, set `INTENT_REPO_ROOT` to its absolute path before running
any cells. The optional `INTENT_DATA_DIR`, `INTENT_OUTPUT_DIR`, and
`INTENT_WORK_DIR` variables redirect data, outputs, and runtime caches.

Generation writes to `data/queries/<generator>/`. The final cell of each
corrected inference notebook exports a canonical CSV under
`data/results/<generator>/`. Raw-result readers prefer a same-stem CSV over
XLSX. Use separate data/output directories for distinct experiment runs.
The import script writes to the repository-local data directory.

## GPU environment

Generation, downstream inference, and local judging require a CUDA-capable
Linux runtime. Colab is supported when the full repository is present and
`INTENT_REPO_ROOT` identifies its location.

Install a PyTorch build compatible with the machine's CUDA runtime, then use
`requirements-generation.txt` for alphabet building and query generation.
Inference and Flow-Judge notebooks create separate `uv`/vLLM environments in
`.work/`. Their setup cells install packages and download weights. Run one model
notebook at a time and execute its cleanup cell before starting the next.
Dependencies and downloaded model/dataset revisions are not fully pinned;
these requirement files are not historical environment lockfiles.

## Authentication

Use `HF_TOKEN` or an existing Hugging Face login for gated models, with access
already granted to the corresponding model repositories. The optional API judge
requires `requirements-openai.txt` and `OPENAI_API_KEY`. Its historical model
identifier can be overridden with `OPENAI_JUDGE_MODEL`; the reasoning setting
uses `OPENAI_JUDGE_REASONING_EFFORT`. That identifier's current availability has
not been verified.

`.env.example` lists the variables and is not loaded automatically. Credentials
belong in runtime environment variables or interactive authentication, not in
committed source cells.

## Experimental limitations

Read `REPRODUCIBILITY.md` before interpreting a resumed or partial run. In
particular, original stochastic generation resumption can change seed indexing;
legacy per-candidate summaries may omit separately logged failures; and the
Mistral/Flow recovery loaders require a complete expected input roster.
