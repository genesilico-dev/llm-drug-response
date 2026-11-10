# Study 08: can LLM knowledge predict tumor response to unscreened compounds?

Written 2026-11-09 before any study-08 API call or outcome. Exploratory: it reuses the
CTRP panel and patient-group cell folds from study 06, which earlier studies inspected.

## Question

Studies 06 and 07 showed that with about 480 labelled cells per drug, ridge learns the
within-drug ranking and LLM gene priors add nothing (+0.0003 Spearman, Qwen and Opus).
Knowledge should matter where labels are absent. Here every test compound is
**unscreened**: it has no response labels at all, and test cells come from unseen
patient groups (double cold start).

Hypothesis: LLM mechanism annotations from compound name and SMILES identify which
screened compounds behave like the new one. A small model trained only on screened
compounds learns when to trust each LLM ("taming"). Together they beat non-LLM
information: general cell sensitivity, chemical structure similarity and CTRP curated
targets.

## Data and splits

- Study-06 frozen CTRP-only inputs (manifest hashes checked): 536 compounds, 720 cells,
  312,760 pairs, `sensitivity = -AUC`, DepMap RNA (19,193 genes). No PRISM.
- Cell folds: study-06 patient-group 3 folds (`cells.fold`).
- Drug folds (primary): 5 folds grouped by structural cluster. Morgan radius-2
  2,048-bit fingerprints; Butina clustering at Tanimoto distance 0.4; clusters shuffled
  with seed 20261109 and assigned greedily to the smallest fold.
- Drug folds (secondary): seeded random 5 folds.
- 15 outer blocks (cell fold x drug fold). A block may use labels only of training
  drugs on training cells.

## Drug profile models

For each cell fold and each of the 536 drugs: kernel ridge on standardized training-cell
RNA (kernel ZZ'/p, standardization from training-cell RNA only, no labels), alpha from
the study-06 grid chosen by inner 3-fold GroupKFold on patient groups. Store out-of-fold
predictions at training cells and predictions at all test cells. A drug's profile is
used only when it is a training drug of the block. Profiles are z-scored with the mean
and SD of their training-cell out-of-fold predictions.

## Methods for a test compound j

All predictions are scores over test cells; the metric is Spearman, so scale is free.

- `general`: mean z-profile of all training drugs (general cell sensitivity).
- `knn_<source>`: weighted mean z-profile of the 10 most similar training drugs with
  similarity > 0; falls back to `general` if none. Sources:
  - `structure`: Morgan Tanimoto.
  - `curated`: Jaccard of CTRP `gene_symbol_of_protein_target` sets.
  - `llm_<model>`: 0.5 x target-gene Jaccard + 0.25 x same mechanism class (excluding
    `other` and `unknown`) + 0.25 x signed-marker Jaccard.
- `marker_<model>`: zero-shot, mean of sign x z-RNA over the LLM's sensitivity markers;
  falls back to `general` if none are valid.
- `tamed_<set>`: ridge (alpha 10, fixed) trained on training drugs x training cells.
  Its features are the method scores above, plus their products with per-drug trust
  descriptors: max structure similarity to training drugs, curated-target
  availability, and each LLM's confidence and `recognized` flag. Training features use
  leave-one-drug-out neighbours among training drugs and out-of-fold profiles. The
  target is per-drug z-scored sensitivity on training cells.
  - `tamed_nonllm`: general, structure, curated.
  - `tamed_<model>`: non-LLM plus one LLM.
  - `tamed_all`: non-LLM plus all LLMs.
  - `tamed_shuffled_<r>`: `tamed_all` with LLM annotations deranged across compounds,
    seeds 0 to 2 (permutation control).

No hyperparameter is tuned on test blocks. K = 10, similarity weights and the stacker
alpha are fixed here.

## LLMs (OpenRouter, provider pinned, no fallbacks, data_collection deny)

| Key | Model | Provider | Role |
|---|---|---|---|
| gemma | google/gemma-4-31b-it | deepinfra/fp8 | small open-weight |
| kimi | moonshotai/kimi-k3.1-pro | moonshotai/mxfp4 | large open-weight |
| ember | fireworks/ember-1 | fireworks | open-weight derivative (Kimi K3) |
| opus | anthropic/claude-opus-5.5 | anthropic | frontier |
| gpt | openai/gpt-5.6-sol | openai | frontier |

Input: compound name and SMILES only. CTRP targets are withheld, so the curated
baseline stays independent. No RNA, labels or cell-line data are sent. Medium reasoning,
reasoning text excluded, strict JSON schema, one generation per compound and up to one
technical retry. Gene symbols are validated against the 19,193 RNA genes. Invalid
symbols are dropped and counted.

Budget: local reservation ceiling of $120 for the study. Per-model ceilings: opus 35,
gpt 35, ember 25, kimi 20, gemma 5. Pilot: 10 compounds per model. A model whose pilot
projects above its ceiling, or whose pilot has more than 2 invalid responses, is
dropped and reported.

Amendment 1 (pilot, before any outcome): requests that end in a transport error (no
HTTP response) do not count as a generation. They are retried up to 4 times with
backoff, and their reservations stay on the ledger. Five Opus pilot compounds hit this.

## Endpoints

Unit: compound. A compound's score is the mean over 3 cell folds of Spearman on observed
test cells. The comparison is a paired bootstrap over compounds, 2,000 draws, seed
20261109.

- **P1 (primary):** `tamed_all - tamed_nonllm`. Success if the mean is at least +0.01,
  the 95% lower bound is above 0, and `tamed_all` beats the mean of the shuffled
  controls with lower bound above 0.
- **P2:** for each LLM, `knn_llm_<model> - knn_structure` and `- knn_curated`.
- Secondary: frontier vs open-weight models; `tamed_all` vs the best single
  `tamed_<model>`; zero-shot `marker_<model>` vs `general`; within-lineage Spearman;
  subsets (probe vs clinical/FDA, LLM-recognized vs not, no curated target);
  random drug folds.

## Limitations declared in advance

LLM pretraining may include CTRP or GDSC findings for named compounds. That is
knowledge, not label leakage from this split, but it means "unscreened in this panel",
not "novel to science". Probe compounds with code names test reasoning from structure.
Folds were used in earlier studies. Multiple secondary contrasts are not adjusted.
