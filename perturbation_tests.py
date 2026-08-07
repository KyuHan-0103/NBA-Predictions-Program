"""Per-feature error-impact diagnostics for the aggregation, split out of
aggregate_team.py.

What lives here
---------------
The two rankings of "which aggregation error actually matters to a prediction":

* perturbation_impact() -- PRIMARY. Perturbs each feature by its own measured
  aggregation MAPE and re-predicts with the fitted ridge model, averaging the
  resulting change in predicted wins over all 30 teams.
* error_impact_deprecated() -- kept only to print alongside it, because the
  disagreement between the two is itself the evidence for using the first (see
  README.md's Test log entry on this).

Same tests, two entry points
----------------------------
Nothing about the tests changed in the move. `python aggregate_team.py` still
runs and prints them exactly as before -- its main() calls
run_perturbation_tests() below -- and `python perturbation_tests.py` runs the
identical tests standalone by rebuilding the same inputs (current-season player
pull -> per-team aggregation -> per-feature MAPE -> ridge fit on prior seasons)
that aggregate_team.main() feeds them.

Import direction: this module imports aggregate_team, not the other way round.
aggregate_team.main() imports run_perturbation_tests() locally, inside the
function, which is what keeps that cycle from existing at import time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aggregate_team import (
    DEF_TARGET,
    GAMES_PER_SEASON,
    TEAM_CSV,
    TEAM_DROP_COLS,
    TEAM_TARGET,
    aggregate_team,
    filter_players,
    load_players,
)
from train_model import build_model, ridge_step


def error_impact_deprecated(mape: pd.Series, coefs: pd.Series) -> pd.DataFrame:
    """DEPRECATED -- see perturbation_impact() for the replacement.

    Crosses per-feature aggregation MAPE with the ridge model's standardized
    coefficients (mape_pct * |coef|) as a proxy for prediction-facing error.
    This is unreliable when features are collinear with offsetting
    coefficients: PACE and POSS correlate at ~0.99 here but carry large
    coefficients of opposite sign (PACE ~-0.97, POSS ~+0.07), so either
    coefficient's magnitude alone overstates how much *that feature's* error
    actually moves a prediction -- some of it is cancelled by the other.
    Kept only so its ranking can be printed alongside perturbation_impact()'s
    for comparison; do not use this to decide which aggregation error matters.

    Second reason not to trust it, added with the logit target: the model now
    fits log-odds, so `coef` is in log-odds per SD while perturbation_impact()
    still reports wins. The two columns are no longer even in convertible
    units -- the logistic's slope depends on where the prediction sits, so
    there's no single multiplier from one to the other. Compare the rankings,
    never the magnitudes.
    """
    out = pd.DataFrame({"mape_pct": mape, "coef": coefs, "abs_coef": coefs.abs()})
    out["impact"] = out["mape_pct"] * out["abs_coef"]
    return out.sort_values("impact", ascending=False)


def perturbation_impact(
    model,
    real_lines: pd.DataFrame,
    mape: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Rank features by how much perturbing them actually moves the model's
    predicted win rate -- a direct replacement for error_impact_deprecated().

    For each feature, and for each of the 30 real teams in `real_lines`, that
    team's real stat line is perturbed by the aggregation's measured MAPE for
    that feature (both +MAPE% and -MAPE%, one feature at a time, all others
    held at their real value), re-predicted with the already-fitted `model`,
    and the change in predicted wins (delta W_PCT * 82) is recorded. The
    table reports, per feature, the mean (over both directions and all 30
    teams) of the absolute win-count change -- this is what "the model
    actually cares" should mean, instead of a standardized-coefficient proxy
    that collinearity can make misleading (see error_impact_deprecated()).

    Now that the model fits log-odds, a feature's win impact is no longer the
    same everywhere: the logistic is steepest at .500 and flattens toward
    either end, so the identical perturbation moves a 41-win team more than a
    60-win team. Averaging over all 30 real teams (as below) is what makes the
    number a league-typical sensitivity rather than one team's; it is not a
    constant that can be applied to a single extreme roster.

    Known limitation: this is a one-at-a-time perturbation, which assumes
    each feature's aggregation error is independent of the others' -- that
    assumption is false here. PACE, POSS, FGA, and PTS errors all originate
    in the same usage-scaling step (see aggregate_team()'s conserve_usage
    scaling), so in practice they move together, not independently, and this
    diagnostic cannot capture that correlated, simultaneous error. Treat it
    as a per-feature sensitivity ranking, not a bound on the aggregation's
    real combined error.
    """
    X = real_lines[feature_cols]
    baseline = model.predict(X)
    rows = []
    for feat in feature_cols:
        frac = mape[feat] / 100.0
        X_plus, X_minus = X.copy(), X.copy()
        X_plus[feat] = X[feat] * (1.0 + frac)
        X_minus[feat] = X[feat] * (1.0 - frac)
        delta_plus = (model.predict(X_plus) - baseline) * GAMES_PER_SEASON
        delta_minus = (model.predict(X_minus) - baseline) * GAMES_PER_SEASON
        mean_abs_win_impact = np.mean((np.abs(delta_plus) + np.abs(delta_minus)) / 2.0)
        rows.append({"feature": feat, "mape_pct": mape[feat], "mean_abs_win_impact": mean_abs_win_impact})
    return pd.DataFrame(rows).set_index("feature").sort_values("mean_abs_win_impact", ascending=False)


def run_perturbation_tests(
    model,
    real_lines: pd.DataFrame,
    mape: pd.Series,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Print both rankings and where they disagree; return (old, new) tables.

    This is the block that used to sit inline in aggregate_team.main(), moved
    here verbatim so both entry points print the same thing.
    """
    # ridge_step() rather than model.named_steps: build_model() wraps the
    # pipeline in a TransformedTargetRegressor for the logit target, so the
    # estimator is one level deeper now. Units are log-odds per SD (see
    # error_impact_deprecated).
    coefs = pd.Series(ridge_step(model).coef_, index=feature_cols)
    impact_old = error_impact_deprecated(mape, coefs)
    impact_new = perturbation_impact(model, real_lines, mape, feature_cols)

    print("\n=== [DEPRECATED] MAPE x ridge coefficient, log-odds units (desc) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(impact_old.to_string())

    print("\n=== Perturbation test: mean abs win impact per feature (desc) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(impact_new.to_string())

    # --- Where the two rankings disagree ---
    rank_cmp = pd.DataFrame({
        "old_rank": impact_old["impact"].rank(ascending=False),
        "new_rank": impact_new["mean_abs_win_impact"].rank(ascending=False),
    })
    rank_cmp["rank_diff"] = (rank_cmp["old_rank"] - rank_cmp["new_rank"]).abs()
    disagreements = rank_cmp.sort_values("rank_diff", ascending=False).head(8)
    print("\n=== Biggest ranking disagreements (old vs. perturbation) ===")
    print(disagreements.to_string())
    print(
        "Expect PACE/POSS to disagree most: their ~0.99 correlation and offsetting\n"
        "coefficients (PACE ~-0.97, POSS ~+0.07 in log-odds per SD) let the deprecated\n"
        "method's coefficient-magnitude proxy overstate/understate impact that the\n"
        "perturbation test measures directly. Note the two tables are now in different\n"
        "units too (log-odds per SD vs. wins) -- compare ranks, not magnitudes."
    )
    return impact_old, impact_new


def build_inputs() -> tuple:
    """Rebuild what the tests need, the same way aggregate_team.main() does.

    Returns (model, real_lines, mape, feature_cols): the ridge model fitted on
    every season EXCEPT the current one (so current-season predictions stay
    out-of-sample), each real team's own current-season line, and the
    aggregation's per-feature mean absolute % error from rebuilding all 30
    teams from their own top-15-by-minutes rosters.

    Standalone use pulls the current season's player stats from the NBA API,
    exactly as aggregate_team.main() does -- there is no cached player CSV for
    the current season in this repo.
    """
    teams_all = pd.read_csv(TEAM_CSV)
    feature_cols = [c for c in teams_all.columns if c not in TEAM_DROP_COLS + [TEAM_TARGET, DEF_TARGET]]
    season = sorted(teams_all["SEASON"].unique())[-1]
    teams = teams_all[teams_all["SEASON"] == season].set_index("TEAM_ID")

    qualified = filter_players(load_players(season))

    errs, agg_ids = [], []
    for tid, grp in qualified.groupby("TEAM_ID"):
        grp = grp.sort_values("MIN", ascending=False).head(15)
        if tid not in teams.index or len(grp) < 5:
            continue
        a = aggregate_team(grp, feature_cols)
        r = teams.loc[tid, feature_cols].astype(float)
        errs.append((100.0 * (a - r) / r.replace(0, np.nan)).abs())
        agg_ids.append(tid)
    mape = pd.concat(errs, axis=1).mean(axis=1)

    train_df = teams_all[teams_all["SEASON"] != season]
    model = build_model()
    model.fit(train_df[feature_cols], train_df[TEAM_TARGET])

    real_lines = teams.loc[agg_ids, feature_cols].astype(float)
    print(f"Season: {season}  |  team features: {len(feature_cols)}  |  "
          f"teams aggregated: {len(agg_ids)}")
    return model, real_lines, mape, feature_cols


def main() -> None:
    model, real_lines, mape, feature_cols = build_inputs()
    run_perturbation_tests(model, real_lines, mape, feature_cols)


if __name__ == "__main__":
    main()
