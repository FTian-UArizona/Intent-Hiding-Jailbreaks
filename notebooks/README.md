# Notebook guide

Start with the CPU analysis, or follow the experimental stages in order.
All paths in this page are relative to this directory.

| Stage | Notebook | Role |
| --- | --- | --- |
| Results | [ASR figures](analysis/asr_figures.ipynb) | Reproduce included ASR tables and figures on CPU |
| Alphabet | [Water filling](alphabet/waterfilling.ipynb) | Fractional prior–posterior matching on the included alphabet |
| Alphabet | [Build task alphabet](alphabet/build_task_alphabet.ipynb) | Rebuild and evaluate the task collection |
| Generation | [Qwen3-32B](generation/qwen3_32b.ipynb) | Generate the primary composed-query pool |
| Generation | [Qwen3-14B](generation/qwen3_14b.ipynb) | Generate queries using the same stored auxiliary bundles |
| Evaluation | [Flow-Judge](evaluation/flow_judge_target_preservation.ipynb) | Response-level target preservation |

## Downstream inference

The directory identifies the query generator; the notebook identifies the
model answering those queries.

| Downstream model | Qwen3-32B queries | Qwen3-14B queries |
| --- | --- | --- |
| Qwen3-1.7B | [Notebook](inference/qwen3_32b/qwen3_1p7b.ipynb) | [Notebook](inference/qwen3_14b/qwen3_1p7b.ipynb) |
| Qwen3-4B | [Notebook](inference/qwen3_32b/qwen3_4b.ipynb) | [Notebook](inference/qwen3_14b/qwen3_4b.ipynb) |
| Qwen3-4B-Instruct-2507 | [Notebook](inference/qwen3_32b/qwen3_4b_instruct.ipynb) | [Notebook](inference/qwen3_14b/qwen3_4b_instruct.ipynb) |
| Qwen3-8B | [Notebook](inference/qwen3_32b/qwen3_8b.ipynb) | — |
| Mistral-7B-Instruct-v0.3 | [Inference](inference/qwen3_32b/mistral_7b.ipynb), then [corrected rejudge](inference/qwen3_32b/mistral_7b_rejudge.ipynb) | — |
| Gemma-3-4B-PT | [Notebook](inference/qwen3_32b/gemma3_4b_pt.ipynb) | — |
| Gemma-3-4B-IT | [Notebook](inference/qwen3_32b/gemma3_4b_it.ipynb) | — |

## Supplementary experiments

- [Gemma-3-27B-PT query generation](optional/gemma3_27b_pt_generation.ipynb)
- [API judge comparison](optional/openai_harmbench_judge.ipynb) — paid API access required

See [experiment setup](../docs/EXPERIMENTS.md) for dependencies, credentials,
input/output paths, and the complete run order.
