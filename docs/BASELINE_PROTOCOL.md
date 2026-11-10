# CTRPv2-native prediction reset

Frozen 2026-11-09 before new model scores. This is a development benchmark, not
independent confirmation: CTRP and DepMap were explored in earlier studies.
Historical studies and the previous abstract remain unchanged.

## Question and data boundary

Can broader CTRPv2 data and more target-drug labels deliver useful prediction
without PRISM? No PRISM files, labels, drug panels, models, graph priors or
prepared study-01 matrices are used. Raw CTRPv2 response data and raw DepMap RNA
and Model metadata are the only biological inputs. The study-01 Python environment
is reused for dependencies, not its pipeline. No patient-benefit claim is made.

## Preparation

- Verify the CTRPv2 archive SHA-256 and publisher member MD5 checksums.
- Use publisher post-QC curves and only SNP-matched-reference experiments.
- Require at least eight fitted concentration points and finite native AUC.
- Use the native master_cpd_id as compound identity; do not merge compounds by
  target or require PRISM overlap. Record names, SMILES and concentration ranges.
- Match CTRP cell names against unique normalized DepMap aliases; exclude ambiguous
  aliases and models without RNA. Aggregate repeat cell-compound AUC by median.
- Endpoint: negative native fitted AUC (higher means more sensitive). No clipping
  or response rescaling across compounds; absolute units differ by assay/range.
- Use PatientID as split group; missing IDs get a unique ModelID-based group.
- Keep compounds with at least 150 mapped patient groups. No eligibility threshold
  on response variation, activity, mechanism, or measured model performance.
- Log every mapping/QC exclusion. Native AUC is not growth-rate corrected and
  curve QC does not guarantee an informative pharmacological response range.
- NCI CTD2 investigative reuse applies; commercial clearance is not asserted.

## Evaluation

Three global patient-group folds: sort unique groups, shuffle with seed 20261109,
then assign round-robin. Each group is tested once. This is same-assay prediction
on unseen patient groups, not new-drug or cross-screen generalization.

Per compound/fold, require 60 training groups and 30 test cells. Both budgets use
the same test cells. Budget 32 selects one observed cell from each of 32 randomly
chosen training groups using seed 20261109 + compound ID + fold. Full uses all
observed training cells. Save all support and test identifiers. Do not filter
test samples by outcomes. Constant observed outcomes give unavailable rho;
constant predictions with variable outcomes receive rho=0 and a status flag.

## Frozen baseline menu

1. Training mean (MAE reference).
2. Training median (MAE reference).
3. Training lineage mean, falling back to training mean.
4. Ridge on all finite, nonconstant RNA genes, standardized on training only.
5. Ridge on the top 1,000 training-variance genes, standardized on training only.
6. Elastic net on the same top 1,000 genes; training labels standardized for fitting.

Three-fold inner GroupKFold chooses hyperparameters by mean validation Spearman;
ties retain the first (more regularized) candidate. Every inner fold independently
fits gene eligibility, variance selection, means and scales. Ridge kernel is
XX'/p with penalties [10, 1, 0.1, 0.01, 0.001]; intercept is unpenalized.
Elastic-net alpha [0.3, 0.1, 0.03], l1_ratio [0.1, 0.5], max_iter=5000, tol=0.001.
Convergence warnings are recorded; incomplete convergence must be disclosed.
Test labels are passed only to scoring, never to fitting or selection.

## Reports and decisions

Report per-drug mean fold Spearman, median drug Spearman, MAE and MAE improvement
against each matched training baseline, panel sizes, warnings and all exclusions.
Do not use pooled drug-cell correlation as a headline. Save predictions, choices,
input/protocol/code hashes and package versions. Resume only matching run signatures.

Primary contrasts: full minus 32 labels within each RNA method; full elastic net
minus full all-gene ridge; full 1,000-gene ridge minus full all-gene ridge. Paired
drug bootstrap, 2,000 draws, seed 20261109, mean folds within drug first. These
intervals are development summaries, conditional on fitted models; drugs share
cells and mechanisms, and intervals are not multiplicity-adjusted.

Smoke: first three eligible native compound IDs, all folds/budgets, for runtime and
code validation only. Full run must include the entire eligible panel, including
smoke compounds. No selecting winners or changing the menu using test outcomes.
If a new model is warranted, record an amendment and treat further reuse of these
folds as exploratory. Independent external validation requires a separate dataset.
No claim of publishability or clinically meaningful performance is guaranteed.