# Study 08 result: LLM knowledge helps predict responses to unscreened compounds

Completed 2026-11-09 under [PROTOCOL.md](PROTOCOL.md) (with amendment 1, made before
any outcome). Outputs: `outputs/evidence.json`, `outputs/annotation_audit.json`.
Exploratory: it reuses the study-06 patient-group cell folds. No PRISM inputs.

## Conclusion

The primary endpoint passes. For compounds with no response labels, tested on tumor
models from unseen patients, a small model that combines LLM mechanism annotations
beats the best non-LLM combination by **+0.0138 Spearman [+0.0104, +0.0175]**. It beats
the same model with LLM annotations shuffled across compounds by **+0.0143
[+0.0107, +0.0180]**. Both pass the pre-set bar (mean of at least +0.01, lower bound above 0).

LLM gene markers used directly are poor (0.08 to 0.13 versus 0.245 for general
sensitivity). LLM output helps when it is used to find which screened compounds a new
compound resembles, with a label-trained model deciding how much to trust it.

## Main table: structure-grouped drug folds (536 compounds, 15 blocks)

Mean over compounds of Spearman on test cells, averaged over three cell folds.

| Method | Spearman |
|---|---:|
| General cell sensitivity | 0.2448 |
| Nearest compounds by structure (Morgan) | 0.2609 |
| Nearest compounds by CTRP curated targets | 0.2693 |
| Small model, non-LLM sources only | 0.2753 |
| Small model + shuffled LLM annotations (mean of 3) | 0.2748 |
| Small model + Gemma 4 31B | 0.2770 |
| Small model + Fireworks Ember-1 | 0.2814 |
| Small model + Kimi K3.1 Pro | 0.2824 |
| Small model + GPT-5.6 Sol | 0.2843 |
| Small model + Claude Opus 5.5 | **0.2910** |
| Small model + three open-weight models | 0.2829 |
| Small model + two frontier models | 0.2906 |
| Small model + all five LLMs (primary) | 0.2891 |
| Reference: ridge with labels for the compound (study 06) | 0.3419 |

The LLM gain closes about 21% of the gap between the non-LLM small model and a
fully labelled compound ((0.2891 - 0.2753) / (0.3419 - 0.2753)).

### Paired contrasts, compound bootstrap 95% intervals

| Contrast | Structure-grouped folds | Random folds |
|---|---:|---:|
| All LLMs minus non-LLM (P1) | +0.0138 [+0.0104, +0.0175] | +0.0152 [+0.0116, +0.0192] |
| All LLMs minus shuffled (P1) | +0.0143 [+0.0107, +0.0180] | +0.0160 [+0.0123, +0.0200] |
| Frontier minus open-weight | +0.0078 [+0.0053, +0.0103] | +0.0079 [+0.0053, +0.0104] |
| All LLMs minus Opus alone | -0.0019 [-0.0034, -0.0007] | -0.0013 [-0.0028, -0.0001] |
| Non-LLM small model minus general | +0.0304 [+0.0228, +0.0385] | +0.0335 [+0.0256, +0.0419] |

Each LLM on its own as a neighbour finder (P2), structure-grouped folds:

| LLM | minus structure kNN | minus curated-target kNN |
|---|---:|---:|
| Gemma 4 31B | -0.0061 [-0.0127, +0.0005] | -0.0145 [-0.0212, -0.0073] |
| Ember-1 | +0.0094 [+0.0020, +0.0168] | +0.0009 [-0.0038, +0.0062] |
| Kimi K3.1 Pro | +0.0093 [+0.0022, +0.0171] | +0.0008 [-0.0042, +0.0065] |
| GPT-5.6 Sol | +0.0082 [+0.0013, +0.0153] | -0.0002 [-0.0062, +0.0061] |
| Claude Opus 5.5 | +0.0183 [+0.0117, +0.0251] | +0.0099 [+0.0046, +0.0157] |

From name and SMILES alone, Opus is the only model that beats curated CTRP targets.

### Within-lineage and subsets (structure-grouped)

- Within-lineage Spearman: non-LLM 0.1062, all LLMs 0.1252, difference **+0.0190
  [+0.0142, +0.0239]**. The gain is not explained by lineage effects.

| Subset | n | All LLMs minus non-LLM |
|---|---:|---:|
| Probe compounds | 336 | +0.0133 [+0.0092, +0.0177] |
| Clinical or FDA | 171 | +0.0175 [+0.0107, +0.0245] |
| No CTRP curated target | 128 | +0.0149 [+0.0068, +0.0233] |
| Recognized by 3 or more LLMs | 426 | +0.0162 [+0.0122, +0.0205] |
| Recognized by 0 to 2 LLMs | 110 | +0.0045 [-0.0007, +0.0099] |

The gain comes mostly from compounds the LLMs say they know. It holds where CTRP has
no curated target, so it is not only the LLMs reproducing CTRP annotations.

### What the small model learned (mean weights over 15 blocks)

The largest weights are LLM neighbour scores gated by the LLM's own confidence (GPT,
Ember) or by neighbour similarity (Opus), and Opus markers gated by confidence.
Gemma's confidence-gated term gets a negative weight, which suggests Gemma is
overconfident. Features are correlated, so individual weights are only indicative.

## Annotation audit (no response labels used)

| Model | Valid / 536 | Recognized | "unknown" class | Curated-target recall | Any curated hit | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 31B (DeepInfra fp8) | 536 | 414 | 111 | 0.446 | 0.473 | $0.38 |
| Kimi K3.1 Pro (Moonshot) | 536 | 440 | 65 | 0.747 | 0.836 | $2.92 |
| Ember-1 (Fireworks) | 536 | 418 | 80 | 0.765 | 0.836 | $3.00 |
| Claude Opus 5.5 (Anthropic) | 536 | 440 | 58 | 0.850 | 0.934 | $6.59 |
| GPT-5.6 Sol (OpenAI) | 536 | 466 | 40 | 0.767 | 0.855 | $12.34 |

Recall is over the 408 compounds that have CTRP targets. Total reported cost **$25.22**,
below the $120 local ceiling. Every compound has a valid annotation from every model.
Opus needed transport retries (amendment 1). Returned model IDs and providers are stored
per attempt in `data/annotations/<model>/`.

## Integrity

- Test check: corrupting labels of test compounds and test cells leaves every
  prediction unchanged (`test_study08.py`, 7 tests pass).
- A smoke run with all LLM annotations blanked gave LLM-arm results equal to the
  non-LLM model (difference under 1e-5).
- Results agree between structure-grouped and random compound folds.
- Input manifest hashes are checked, and the code, protocol and annotation digests are
  stored in `evidence.json`.

## Limitations

- The LLMs may have seen CTRP, GDSC or literature results for named compounds during
  pretraining. They are asked for mechanism, not responses, but "unscreened" means
  unscreened in this panel, not new to science. The smaller gain on compounds the LLMs
  do not recognize (+0.0045, interval includes 0) fits this.
- One panel (CTRP v2), one generation per model, and a fixed prompt and weighting. The
  cell folds were inspected in earlier studies.
- 451 structural clusters for 536 compounds, so structure-grouped folds are only
  slightly stricter than random.
- Secondary contrasts are not adjusted for multiplicity.
- Next validation step: an external screen (for example GDSC or PRISM compounds absent
  from CTRP), and compounds released after the models' training cutoffs.
