"""Within-lineage diagnostic extracted from the original evaluation."""

import numpy as np
import pandas as pd
from benchmark import rho


def lineage_scores(predictions, cells):
    joined = predictions.merge(
        cells[["lineage", "group"]],
        left_on="cell_id",
        right_index=True,
        validate="many_to_one",
    )
    strata = []
    for (drug, fold, method, lineage), part in joined.groupby(
        ["drug_id", "fold", "method", "lineage"]
    ):
        reason = "usable"
        if lineage == "unknown":
            reason = "unknown_lineage"
        elif len(part) < 10 or part.group.nunique() < 5:
            reason = "too_small"
        elif np.ptp(part.observed) == 0:
            reason = "constant_observed"
        score = (
            rho(part.observed.to_numpy(), part.predicted.to_numpy())
            if reason == "usable"
            else np.nan
        )
        strata.append(
            {
                "drug_id": drug,
                "fold": fold,
                "method": method,
                "lineage": lineage,
                "n": len(part),
                "groups": part.group.nunique(),
                "status": reason,
                "rho": score,
            }
        )
    return pd.DataFrame(strata)
