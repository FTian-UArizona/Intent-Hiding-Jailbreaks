# Validation

Validation date: 2026-09-25.

| Check | Result |
| --- | --- |
| Public notebook source | 19 notebooks, 249 code cells: static Python checks passed; saved outputs and execution counts cleared |
| IPython statements | 109 shell/magic statements explicitly excluded from Python compilation; not executed |
| Credential-pattern scan | No matching hard-coded credential literals in the public package; the source archive contained an API key that was removed |
| Scientific source comparison | Original prompt/template strings, model IDs, decoding settings, seeds, and function bodies preserved across the 16 alphabet/generation/inference notebooks before path-only normalization and output exports |
| Water-filling notebook | Executed completely on CPU against the included 100-target/50-auxiliary alphabet; prior approximately 0.103113359969 |
| Figure reproduction | Original notebook and reusable CLI executed on CPU; all three ASR CSVs and all three PNGs are byte-identical to the original outputs |
| Figure validation cases | Missing labels treated as failures; 100-target denominator retained; invalid nonbinary labels rejected |
| Query format compatibility | Original 32B CSV/XLSX copies: identical 10,000 rows and 23 columns; matched 14B/Gemma inputs contain 10,000 rows each |
| Generation/inference handoffs | 14/14 CPU input checks passed: three generation stages built 10,000 jobs; ten inference loaders built 10,100 evaluation rows; corrected Mistral rejudge accepted 10,100 staged responses |
| Complete 14B source recovery | All 38 fields of 10,099 shared rows match exactly; alternate original workbook supplies one additional direct baseline |
| Flow-Judge source validation | All 80,800 rows across eight conditions passed the strict input loader using the complete 14B source |
| Judge parsing | Pure Flow/OpenAI parsing routines checked on CPU without model or API calls |
| Archive import | All 14 manifest artifacts verified against their recorded sizes and SHA-256 hashes |

The CPU harness supplied a no-op notebook `display` function and ordinary
pandas/numpy globals where setup cells normally provide them. Mistral rejudge
used a file-selection adapter pointing to an archived result workbook because
new GPU inference outputs had not been generated. This tests its loader,
not the Colab upload interface.

Not executed: GPU inference, query generation, model or dataset downloads,
package installation cells, vLLM servers, live authentication, paid API calls,
or full checkpoint recovery. Exact compatibility of new CUDA/package versions
and model availability remain runtime checks for the user's environment.

The static checker implements lightweight notebook/source checks with the
Python standard library. It is not a full nbformat schema validator, dependency
resolver, or credential-history scanner. The GitHub Actions workflow has not been run on GitHub; its CPU checks were
executed locally.

## Public data and reusable analysis

The public figure command aggregates 101,000 raw candidate outcomes across ten
conditions into 6,000 rows, retaining each target's candidate and success counts
at every size. It reproduces the original ASR and incremental-gain tables without
loading raw query or response text. The analysis notebook now calls the same
Python functions as the command-line entry point.

The regression suite compares all three exported tables against the archived
references and checks that changing raw 14B direct outcomes cannot alter the
corrected generator-comparison curves. Numeric comparisons allow only binary
floating-point serialization roundoff (absolute tolerance 1e-12).

All 28 ordinary supplemental notebook/CSV/XLSX uploads matched archived content
exactly. Import from the separately uploaded filenames validated all ten raw
result artifacts. The separate Flow-Judge archive contains 80,800 valid judgments;
the earlier standalone checkpoint has 216 unresolved rows. Source hashes, shared
judgments, normalized levels, and rubric identity were compared; see
`FLOW_RESULTS.md` for the detailed audit.
