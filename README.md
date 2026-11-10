# LLM pharmacological knowledge for cold-start drug response

Research code for **Integrating language model-derived pharmacological knowledge
with transcriptomic models for cold-start drug response prediction**.

Authors: Rangan Das, K Maheswari, Debarka Sengupta.

The experiment holds out both compound groups and patient groups in CTRPv2.
Separate RNA ridge models predict each reference drug's response. A small ridge
meta-model combines transferred response profiles using structure, curated targets,
and LLM annotations of compound names and SMILES. Only names and SMILES are sent to
the annotation provider, not expression, response labels or cell identifiers.

## Findings and scope

Across 536 compounds, 720 cell lines and 312,760 measurements, the structure-grouped
benchmark's mean per-drug Spearman correlation increased from **0.2753 to 0.2891**
with all five LLM annotations. The paired-drug bootstrap difference was
**0.0138 [0.0104, 0.0175]**. Scores rank held-out cell lines separately for each drug,
average over its cell folds, then weight all drugs equally; they are not a pooled
correlation across drug–cell pairs. These are historical, exploratory results, not
newly rerun estimates from this export.

This repository includes preparation, the numerical baseline, annotation,
cold-start evaluation, tests, protocols and aggregate evidence. It excludes raw
datasets, cell-level predictions, annotation API payloads, credentials, logs,
unrelated studies and the original repository's Git history. No MCP server or
agentic tool-selection system is included.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then from this directory:

```sh
uv sync --locked
uv run python -m pytest
uv run python scripts/check_results.py
```

Tests and aggregate-result checks need no credentials, raw datasets or paid calls.
Python 3.12 and the recorded numerical dependency versions are pinned in `uv.lock`.
The project is a standalone script-based research repository, not an installed library.

## Data and experiment reproduction

See [DATA.md](docs/DATA.md) for required filenames, source links and SHA-256 hashes.
Place inputs under `data/raw` or set the **shell environment variable** `DRP_RAW_DIR`
to a directory with the same layout. This variable is not read from `.env`.

```sh
uv run python scripts/prepare.py
uv run python scripts/evaluate.py profiles
```

Preparation writes `data/prepared`; profiles and annotations use `data/experiment`.
Profile fitting can be expensive. Existing profile files are reused by filename:
delete the three `data/experiment/profiles_fold*.npz` files whenever inputs, ordering,
code or dependencies change. This release preserves that historical cache behavior.

For annotation, set `OPENROUTER_API_KEY` in your shell or in a local ignored `.env`
(template: `.env.example`). **The following commands make paid external API calls.**
Review `MODELS` and the prompt in `scripts/annotate.py` first. Model identifiers,
providers and prices are historical configurations, not guarantees of availability.

```sh
uv run python scripts/annotate.py gemma
uv run python scripts/annotate.py kimi
uv run python scripts/annotate.py ember
uv run python scripts/annotate.py opus
uv run python scripts/annotate.py gpt
uv run python scripts/audit.py
uv run python scripts/evaluate.py evaluate
```

Per-model ledgers reserve estimated request costs against configured local ceilings
(sum $120). These are not a provider-side billing guarantee. Use a provider-side key
limit too. Do not run multiple annotation processes for the **same** model or delete
ledgers to bypass limits. Reruns are nondeterministic; full historical LLM responses
are not distributed, so bit-for-bit end-to-end reproduction is not promised.
The evaluator permits missing annotations; inspect the audit and evidence coverage
before comparing results (the historical run had 536 valid annotations per model).

`scripts/benchmark.py --help` describes the optional labeled-drug numerical baseline.
Regenerated outputs go to ignored `outputs/`, never overwrite the frozen `results/`.

## Interpretation and limitations

- Development-fold reuse and unadjusted confidence intervals make this exploratory.
- Held-out drugs are not necessarily unknown to science or to model pretraining.
  Pretraining exposure to benchmark information cannot be excluded.
- Butina structure groups do not guarantee a strict maximum cross-fold similarity;
  native compound IDs can include mixtures.
- RNA normalization and hyperparameter selection are not fully nested within all
  reference-profile inner folds. Outer test outcomes are excluded, but tests do not
  establish absence of every possible leakage pathway.
- Stacker score columns are standardized separately in train and test cell batches.
  This uses test covariate distributions (not test outcomes): a transductive batch
  evaluation, not an independently calibrated single-sample deployment model.
- Cached profiles and exported evidence have incomplete dependency provenance.
  Clear caches when changing the experiment; see the provenance note below.
- Curated target agreement is not biological ground truth. Opus neighbor transfer
  also improved without stacking; a claim that stacking is always necessary is not
  supported. Coefficient signs are not causal explanations of model reliability.
- Prospective novel-compound, external cohort and patient validation remain open.
  **Not for clinical decision-making.**

## Evidence and provenance

`results/` contains historical summary JSON and per-drug aggregate CSVs. The frozen
code/annotation/protocol hashes describe the original experiments, not the relocated
scripts here. Preparation and baseline came from study 06; annotation/evaluation
from study 08; a within-lineage helper was extracted from study 06. Original source
files were uncommitted at export, so a historical Git commit alone does not identify
them. This release starts fresh history and changes paths/imports, credential
loading, packaging and checks; it does not claim a fresh full-data replication.
`docs/HISTORICAL_RESULT.md` retains original study names and paths for context:
historical `outputs/` corresponds to bundled `results/`, `test_study08.py` to
`tests/test_pipeline.py`, and historical `data/annotations` to the current
`data/experiment/annotations`. Refer to this README for current usage and caveats.

## Rights and acknowledgments

Public visibility does **not** grant an open-source license. No new software license
has been assigned by this extraction; contact GeneSilico for reuse permission.
CTRP, DepMap, model outputs and model weights retain their own terms. No commercial
data-use permission or model-weight redistribution right is asserted. See
[DATA.md](docs/DATA.md). GitHub Copilot assisted with analysis code and drafting;
authors remain responsible for verification.