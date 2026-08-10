"""Estimate a 5-man lineup's DEF_RATING from its five players' *prior-season*
individual stats.

Why this exists
---------------
README.md's largest documented limitation is that DEF_RATING is dropped from
the fictional-roster path: it is opponent-dependent, and every synthesis route
tried before (a box-score sub-model, a personal composite, an awards composite)
either failed outright or "worked" only by borrowing the real, current-season,
on-court DEF_RATING of the very team being predicted -- circular. This model
attacks the same problem under a construction that makes that circularity
*causally impossible*, and reports how much honest signal is left.

The two design rules that make it non-circular
----------------------------------------------
1. STRICT ONE-SEASON LAG. A 2023-24 lineup is described only by its players'
   2022-23 stats. A feature computed a year before the target was observed
   cannot contain the target.
2. NO ON-COURT TEAM CONTEXT, EVER, FROM ANY SEASON. Player DEF_RATING,
   OFF_RATING, NET_RATING, PLUS_MINUS and any on/off split are forbidden --
   they describe the team-with-him-in-it, not the player, which is exactly the
   circularity above. What is permitted: the player's own box score (BLK, STL,
   DREB, OREB, PF, PFD, BLKA), his own rebound rates, closest-defender
   tracking (opponent FG% on shots he defended), hustle stats, height, weight,
   position, MIN, GP, AGE. PIE and USG_PCT are also NOT used: PIE mixes both
   teams' game totals into its denominator and USG_PCT is a share of team
   possessions used while on court, so neither is cleanly "his own".

Data
----
* Target/weights: lineup_season_stats.csv (pull_lineups.py) -- each real 5-man
  lineup's own observed DEF_RATING, weighted by the season-total possessions
  behind it (a 40-possession lineup's rating is mostly noise; see
  label_noise_decomposition()).
* Features: player_all_seasons.csv (pull_player_seasons_all.py) for the box
  score + rebound rates, player_defense_stats.csv (pull_player_defense.py) for
  bio / closest-defender tracking / hustle -- both looked up at SEASON - 1.
* Evaluation: rolling-origin (expanding-window) CV by season -- each fold trains
  on every season before its single test season, never shuffled or sampled
  randomly (see rolling_origin_folds()). Headline metric is the season-centered
  target, because league-average DEF_RATING rose ~10 points across the sample
  and a raw-target R2 is mostly being charged for that drift.

Permutation invariance
----------------------
Five players are an unordered set, so slot-wise concatenation would make the
same lineup in a different order a different input. Features are pooled instead
(mean, plus max/min/std on the columns where "one elite rim protector" is the
hypothesis). Two notes on doing that honestly:
  * At a FIXED group size of 5, a sum is exactly 5x the mean, so including both
    would make the design matrix exactly singular. Only the mean is kept.
  * Pooling is mathematically order-free but not automatically *bit*-identical:
    floating-point addition isn't associative, so a shuffled mean can differ in
    the last bit. build_lineup_features() therefore sorts each lineup's players
    by PLAYER_ID before aggregating, which makes invariance exact.
test_permutation_invariance() shuffles each lineup's five slots and asserts the
feature matrix is byte-identical.

No-prior-season players
-----------------------
A lineup is DROPPED if any of its five players lacks a usable prior season
(rookies; also a prior season too small to be a stable line, reusing
aggregate_team.py's GP/MPG eligibility). Imputing a league-average defender for
a rookie would inject a fabricated feature vector into a row whose label is
real, which is worse than a smaller sample. The cost is a documented sample
bias toward veteran lineups (see README.md).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aggregate_team import (
    LINEUP_CSV,
    LINEUP_PLAYER_ID_COLS,
    MIN_GP,
    MIN_MPG,
    PLAYER_SEASONS_CSV,
    TEAM_CSV,
)
PLAYER_DEFENSE_CSV = "player_defense_stats.csv"
TARGET = "DEF_RATING"

# Ridge penalty grid -- deliberately NOT train_model.py's ALPHAS
# (logspace(-3, 3, 25)), for a reason worth recording because it silently
# broke the ridge fit here:
#
# sklearn's ridge objective is  sum_i w_i (y_i - x_i.b)^2 + alpha ||b||^2, so
# the sample weights and the penalty share one scale. Weights were being passed
# as raw season-total possessions (mean ~550 at the primary floor), which
# multiplies the data term by ~550 and therefore divides the effective penalty
# by ~550: alpha = 1000 was really acting like alpha ~= 2. RidgeCV then pinned
# to the top of train_model.py's grid, and "Ridge and OLS agree to three
# decimals" was an artifact of an unregularized fit, not evidence that the
# signal is robust to regularization.
#
# Fix: fit_models() mean-normalizes the weights (relative weighting, and so
# every possession-weighted metric, is unchanged), and the grid runs to 1e6 so
# the selected alpha can be checked for sitting strictly inside it.
LINEUP_ALPHAS = np.logspace(-3, 6, 40)

# Possession floors compared in the sensitivity analysis (main()). No single
# floor is assumed correct -- sample size, label noise and held-out accuracy
# trade off against each other, and main() prints that trade before anything is
# chosen from it.
POSS_FLOORS = (50, 100, 200, 250)
# Chosen FROM that sweep, not before it, and re-picked when the evaluation moved
# to rolling-origin CV. Under the CV folds (6 held-out seasons, season-centered
# target, Ridge best at every floor):
#   floor  lineup R2 [95% CI]        share of ceiling  team slope  team R2  team MAE vs league avg
#     50   +0.026 [+0.016, +0.037]        0.158           0.716     0.218      2.24 vs 2.23 (worse)
#    100   +0.035 [-0.003, +0.073]        0.153           0.712     0.224      2.17 vs 2.19
#    200   +0.058 [-0.005, +0.120]        0.202           0.708     0.234      2.05 vs 2.11
#    250   +0.058 [-0.006, +0.122]        0.176           0.705     0.240      1.96 vs 2.07
# 250 wins on the metric the estimate would actually be used through (team-level
# R2 and calibrated team MAE, the only floor a clear margin under the
# league-average baseline) and ties 200 on lineup-level R2. Two honest caveats,
# both carried in README.md: (1) above floor 50 the lineup-level CI includes
# zero -- only floor 50's small effect is separable from zero across folds, and
# it is the floor whose team-level estimate is no better than the league
# average; (2) a high floor leaves fewer team-seasons with any qualifying lineup
# (133 of 180 vs 179 at floor 50), so its calibration is fit on a self-selected
# set of stable rotations. Floor 200 is statistically indistinguishable with 42%
# more held-out lineups and is the conservative alternative.
PRIMARY_POSS_FLOOR = 250

# Hustle stats only exist from 2016-17 (see pull_player_defense.py), so using
# them costs the three oldest lineup seasons (a lineup needs its players'
# PRIOR season). Off by default; main() reports the with/without comparison on
# the identical reduced sample so the trade is measured, not assumed.
HUSTLE_START_SEASON = "2016-17"

# --- Feature families -------------------------------------------------------
# Volume stats put on a per-36-minute basis: a 12-mpg backup big and a 34-mpg
# starter should be compared on rate, not on how much his coach played him
# (playing time enters separately, as MIN/GP levels below).
RATE_36_COLS = ["BLK", "STL", "DREB", "OREB", "PF", "PFD", "BLKA", "D_FGA"]
# Already rates/levels -- used as-is. NORMAL_FG_PCT is deliberately excluded:
# PCT_PLUSMINUS == D_FG_PCT - NORMAL_FG_PCT exactly, so all three together
# would be perfectly collinear.
LEVEL_COLS = [
    "DREB_PCT", "OREB_PCT", "D_FG_PCT", "PCT_PLUSMINUS",
    "MIN", "GP", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "AGE",
]
HUSTLE_36_COLS = [
    "CONTESTED_SHOTS", "CONTESTED_SHOTS_2PT", "CONTESTED_SHOTS_3PT",
    "DEFLECTIONS", "CHARGES_DRAWN", "DEF_BOXOUTS", "DEF_LOOSE_BALLS_RECOVERED",
]
# Order statistics (max/min/std across the five) on the columns where a mean
# would hide the thing that plausibly matters most -- one elite rim protector,
# one elite perimeter disruptor, one player who can be attacked.
ORDER_STAT_COLS = ["BLK_36", "STL_36", "DREB_PCT", "PLAYER_HEIGHT_INCHES"]
# Raw per-game columns kept only to build the naive box-score baseline.
NAIVE_BOX_COLS = ["BLK", "STL", "DREB"]
# Position buckets counted per lineup. G is dropped, not forgotten: the three
# counts always sum to 5, so keeping all three would be collinear with the
# intercept. "2 bigs" is then read as N_CENTER/N_FORWARD against a guard base.
POSITION_GROUPS = ["C", "F", "G"]
POSITION_FEATURES = ["N_CENTER", "N_FORWARD"]
# Roster continuity, from PRIOR-season team membership only: of the 10 pairs
# among five players, the fraction who were on the same team last season. It is
# a property of the roster, not of any current-season performance -- no
# minutes-together, no on-court results, nothing dated after the roster is
# known -- so it is computable before a season starts and for a hypothetical
# roster (five players who never shared a team score 0.0, which is exactly the
# fictional-roster case). Added because the team-change diagnostic found
# continuity lineups defending ~1.5 points better than their players' own stats
# predict and reshuffled ones ~1.1 points worse: an effect the model was blind
# to, being paid for by biased predictions.
CONTINUITY_FEATURES = ["CONTINUITY_PAIR_FRAC"]
N_PAIRS = 10  # C(5, 2)

GBM_PARAMS = dict(random_state=0)  # sklearn defaults otherwise -- see fit_models()

# Short labels for the floor-sensitivity summary table (the full row names are
# too wide to put side by side).
CONST_BASELINE = "Baseline: constant (league avg)"
NAIVE_BASELINE = "Baseline: BLK+STL+DREB sum"
ROW_LABELS = {CONST_BASELINE: "const", NAIVE_BASELINE: "naive", "OLS": "OLS", "Ridge": "Ridge", "GBM": "GBM"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def prior_season(season: str) -> str:
    """'2023-24' -> '2022-23'."""
    start = int(season.split("-")[0]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def load_player_features(use_hustle: bool = False) -> pd.DataFrame:
    """One row per (PLAYER_ID, SEASON) of purely individual descriptors.

    Marker columns (_ELIGIBLE, _HAS_DEFENSE, _COMPLETE) are carried through so
    build_lineup_features() can attribute each dropped lineup to a reason
    instead of silently losing it.
    """
    box = pd.read_csv(PLAYER_SEASONS_CSV)
    defense = pd.read_csv(PLAYER_DEFENSE_CSV)

    df = box.merge(
        defense.drop(columns=["PLAYER_NAME"]),
        on=["PLAYER_ID", "SEASON"],
        how="left",
        validate="one_to_one",
        indicator="_defense_merge",
    )
    df["_HAS_DEFENSE"] = df["_defense_merge"] == "both"
    # Same eligibility filter the aggregation contract requires of a roster:
    # a 3-game, 4-mpg line is not a stable description of a player.
    df["_ELIGIBLE"] = (df["GP"] >= MIN_GP) & (df["MIN"] >= MIN_MPG)

    per_36 = 36.0 / df["MIN"].where(df["MIN"] > 0)
    rate_cols = RATE_36_COLS + (HUSTLE_36_COLS if use_hustle else [])
    for c in rate_cols:
        df[f"{c}_36"] = df[c] * per_36

    df["POSITION_GROUP"] = df["PLAYER_POSITION"].str.split("-").str[0]
    df.loc[~df["POSITION_GROUP"].isin(POSITION_GROUPS), "POSITION_GROUP"] = np.nan

    pool_cols = pooled_columns(use_hustle)
    df["_COMPLETE"] = (
        df["_ELIGIBLE"]
        & df["_HAS_DEFENSE"]
        & df["POSITION_GROUP"].notna()
        & df[pool_cols].notna().all(axis=1)
    )

    keep = (
        ["PLAYER_ID", "SEASON", "PLAYER_NAME", "TEAM_ID"]
        + pool_cols + NAIVE_BOX_COLS + ["POSITION_GROUP"]
        + ["_ELIGIBLE", "_HAS_DEFENSE", "_COMPLETE"]
    )
    out = df[keep].rename(columns={"TEAM_ID": "PRIOR_TEAM_ID", "SEASON": "PRIOR_SEASON"})
    return out


def pooled_columns(use_hustle: bool = False) -> list[str]:
    """Per-player columns that get pooled across the five (mean)."""
    rate = RATE_36_COLS + (HUSTLE_36_COLS if use_hustle else [])
    return [f"{c}_36" for c in rate] + LEVEL_COLS


def feature_columns(use_hustle: bool = False) -> list[str]:
    """The exact feature shape the lineup model consumes."""
    pooled = [f"{c}_mean" for c in pooled_columns(use_hustle)]
    orders = [f"{c}_{stat}" for c in ORDER_STAT_COLS for stat in ("max", "min", "std")]
    return pooled + orders + POSITION_FEATURES + CONTINUITY_FEATURES


def load_lineups(poss_floor: float, min_season: str | None = None) -> pd.DataFrame:
    """Real 5-man lineups clearing `poss_floor` season-total possessions.

    The floor is on season-total possessions (POSS is stored per-game by
    pull_lineups.py), the same axis aggregate_team.py's LINEUP_POSS_FLOOR uses.
    """
    lineups = pd.read_csv(LINEUP_CSV)
    lineups["POSS_TOTAL"] = lineups["POSS"] * lineups["GP"]
    lineups = lineups.loc[lineups["POSS_TOTAL"] >= poss_floor]
    if min_season is not None:
        lineups = lineups.loc[lineups["SEASON"] >= min_season]
    return lineups.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Permutation-invariant feature construction
# --------------------------------------------------------------------------- #
def prior_continuity(long: pd.DataFrame) -> pd.DataFrame:
    """Fraction of the 10 player pairs who shared a team the PRIOR season.

    Computed from prior-season TEAM_ID alone: for each lineup, count the pairs
    whose two players' prior-season teams match, over the 10 possible pairs. If
    a lineup's five players came from k distinct prior teams with group sizes
    n_1..n_k, that count is sum(n_j choose 2) -- so it's a closed form over the
    group-size counts, no pair enumeration needed, and it is permutation-
    invariant for the same reason a value_counts() is.

    Deliberately NOT used: anything about how those players actually performed
    or how many minutes they shared. This has to be knowable for a roster that
    has never played (see CONTINUITY_FEATURES), so only membership counts.

    Caveat inherited from the data: player_all_seasons.csv keeps only a traded
    player's final-team stint per season (pull_player_seasons_all.py), so "same
    team last season" means "ended last season on the same team".
    """
    sizes = long.groupby(["_row", "PRIOR_TEAM_ID"]).size()
    pairs = (sizes * (sizes - 1) / 2).groupby(level="_row").sum()
    return pd.DataFrame({"CONTINUITY_PAIR_FRAC": pairs / N_PAIRS})


def build_lineup_features(
    lineups: pd.DataFrame,
    players: pd.DataFrame,
    use_hustle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Pool each lineup's five players' prior-season stats into one feature row.

    Returns (X, meta, summary):
      * X    -- features in feature_columns() order, one row per kept lineup.
      * meta -- SEASON, TEAM_ID, the target, the possession weight, the naive
                baseline's single column, and the team-change flags the leakage
                diagnostic needs.
      * summary -- how many lineups were considered/kept, and one count per
                drop reason (never a silent drop).

    Bit-exact invariance: each lineup's five players are sorted by PLAYER_ID
    before aggregating, so any input ordering produces the identical floating-
    point reduction order (see module docstring).
    """
    pool_cols = pooled_columns(use_hustle)

    lu = lineups.reset_index(drop=True).copy()
    lu["_row"] = np.arange(len(lu))
    lu["PRIOR_SEASON"] = lu["SEASON"].map(prior_season)

    long = lu[["_row", "PRIOR_SEASON", "TEAM_ID"]].join(lu[LINEUP_PLAYER_ID_COLS]).melt(
        id_vars=["_row", "PRIOR_SEASON", "TEAM_ID"],
        value_vars=LINEUP_PLAYER_ID_COLS,
        var_name="_slot",
        value_name="PLAYER_ID",
    )
    long["PLAYER_ID"] = long["PLAYER_ID"].astype("int64")
    long = long.merge(players, on=["PLAYER_ID", "PRIOR_SEASON"], how="left", validate="many_to_one")

    # --- Drop accounting: attribute each incomplete lineup to one reason ------
    has_row = long["_COMPLETE"].notna().to_numpy()
    eligible = long["_ELIGIBLE"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    has_def = long["_HAS_DEFENSE"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    is_complete = long["_COMPLETE"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    long["_ok"] = is_complete
    long["_no_row"] = ~has_row
    long["_too_small"] = has_row & ~eligible
    long["_no_tracking"] = eligible & ~has_def
    long["_incomplete_other"] = eligible & has_def & ~is_complete

    by_row = long.groupby("_row")[["_no_row", "_too_small", "_no_tracking", "_incomplete_other"]].any()
    complete = long.groupby("_row")["_ok"].all()

    summary = {
        "lineups_considered": len(lu),
        "lineups_used": int(complete.sum()),
        "dropped_no_prior_season_row": int((~complete & by_row["_no_row"]).sum()),
        "dropped_prior_season_too_small": int(
            (~complete & ~by_row["_no_row"] & by_row["_too_small"]).sum()
        ),
        "dropped_no_tracking_row": int(
            (~complete & ~by_row["_no_row"] & ~by_row["_too_small"] & by_row["_no_tracking"]).sum()
        ),
        "dropped_missing_feature_value": int(
            (~complete & ~by_row["_no_row"] & ~by_row["_too_small"] & ~by_row["_no_tracking"]
             & by_row["_incomplete_other"]).sum()
        ),
    }

    kept_rows = complete.index[complete.to_numpy()]
    use = long[long["_row"].isin(kept_rows)]
    # Canonical order -> bit-identical pooling regardless of input slot order.
    use = use.sort_values(["_row", "PLAYER_ID"], kind="mergesort")

    grouped = use.groupby("_row", sort=True)
    means = grouped[pool_cols].mean().add_suffix("_mean")
    orders = grouped[ORDER_STAT_COLS].agg(["max", "min", "std"])
    orders.columns = [f"{c}_{stat}" for c, stat in orders.columns]

    counts = (
        pd.crosstab(use["_row"], use["POSITION_GROUP"])
        .reindex(columns=POSITION_GROUPS, fill_value=0)
        .astype("int64")
    )
    positions = pd.DataFrame(
        {"N_CENTER": counts["C"], "N_FORWARD": counts["F"]}, index=counts.index
    )
    continuity = prior_continuity(use)

    X = pd.concat([means, orders, positions, continuity], axis=1)[feature_columns(use_hustle)]

    same_team = (use["PRIOR_TEAM_ID"] == use["TEAM_ID"]).groupby(use["_row"]).all()
    naive = grouped[NAIVE_BOX_COLS].sum().sum(axis=1)  # BLK + STL + DREB, summed over the five

    meta = lu.set_index("_row").loc[kept_rows, ["SEASON", "TEAM_ID", TARGET, "POSS_TOTAL"]].copy()
    meta["ALL_SAME_TEAM"] = same_team
    meta["NAIVE_BOX_SUM"] = naive

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("lineup features contain non-finite values")
    if not X.index.equals(meta.index):
        raise ValueError("feature/meta rows are misaligned")
    return X, meta, summary


def test_permutation_invariance(
    lineups: pd.DataFrame, players: pd.DataFrame, use_hustle: bool = False, seed: int = 0
) -> None:
    """Shuffle the five player slots within every lineup row and assert the
    feature matrix comes back byte-identical (not merely close).

    Raises AssertionError on any difference -- this is a correctness gate on
    the permutation-invariance claim, not a printout.
    """
    X_ref, meta_ref, _ = build_lineup_features(lineups, players, use_hustle)

    rng = np.random.default_rng(seed)
    shuffled = lineups.copy()
    ids = shuffled[LINEUP_PLAYER_ID_COLS].to_numpy()
    ids = rng.permuted(ids, axis=1)  # independent shuffle per row
    shuffled[LINEUP_PLAYER_ID_COLS] = ids
    X_shuf, meta_shuf, _ = build_lineup_features(shuffled, players, use_hustle)

    assert list(X_ref.columns) == list(X_shuf.columns), "column order changed under permutation"
    assert X_ref.shape == X_shuf.shape, f"shape changed: {X_ref.shape} vs {X_shuf.shape}"
    a, b = X_ref.to_numpy(dtype=float), X_shuf.to_numpy(dtype=float)
    assert a.tobytes() == b.tobytes(), "feature matrix is not byte-identical under permutation"
    assert (meta_ref[TARGET].to_numpy() == meta_shuf[TARGET].to_numpy()).all(), "rows misaligned"
    n_moved = int((lineups[LINEUP_PLAYER_ID_COLS].to_numpy() != ids).any(axis=1).sum())
    print(f"Permutation invariance: PASS -- byte-identical features for {len(X_ref)} lineups "
          f"({n_moved} of {len(lineups)} rows actually reordered)")


# --------------------------------------------------------------------------- #
# Label noise: what accuracy is even achievable at a given possession floor
# --------------------------------------------------------------------------- #
def label_noise_decomposition(floors=POSS_FLOORS) -> pd.DataFrame:
    """Split the observed spread in lineup DEF_RATING into true spread + noise.

    A lineup's DEF_RATING is a sample mean over its possessions, so its
    sampling variance falls like 1/POSS while the real, between-lineup spread
    does not. Binning every lineup-season by possessions and fitting

        Var_observed(bin) = Var_true + k * mean(1 / POSS_total)

    across bins (weighted by bin size) recovers both pieces: `k` is the
    per-possession noise scale, Var_true the irreducible spread. Season means
    are removed first, because league-average DEF_RATING drifts ~10 points
    across this sample (see main()'s era note) and that drift is neither noise
    nor between-lineup spread.

    The per-floor "R2 ceiling" (Var_true / Var_observed) is the highest R2 any
    model could score at that floor even if it predicted each lineup's true
    defensive quality exactly -- the number that makes a reported R2 readable.
    """
    lineups = pd.read_csv(LINEUP_CSV)
    lineups["POSS_TOTAL"] = lineups["POSS"] * lineups["GP"]
    y = lineups[TARGET] - lineups.groupby("SEASON")[TARGET].transform("mean")

    bins = pd.qcut(lineups["POSS_TOTAL"], 12, duplicates="drop")
    per_bin = pd.DataFrame({
        "var": y.groupby(bins, observed=True).var(),
        "inv_poss": (1.0 / lineups["POSS_TOTAL"]).groupby(bins, observed=True).mean(),
        "n": y.groupby(bins, observed=True).size(),
    }).dropna()
    slope, intercept = np.polyfit(per_bin["inv_poss"], per_bin["var"], 1, w=np.sqrt(per_bin["n"]))
    var_true = max(intercept, 0.0)

    rows = []
    for floor in floors:
        sel = lineups["POSS_TOTAL"] >= floor
        var_obs = float(y[sel].var())
        noise_var = float(slope * (1.0 / lineups.loc[sel, "POSS_TOTAL"]).mean())
        rows.append({
            "poss_floor": floor,
            "lineups": int(sel.sum()),
            "mean_poss": float(lineups.loc[sel, "POSS_TOTAL"].mean()),
            "sd_observed": np.sqrt(var_obs),
            "sd_noise_implied": np.sqrt(noise_var),
            "sd_true_implied": np.sqrt(max(var_obs - noise_var, 0.0)),
            "noise_share_of_var": noise_var / var_obs if var_obs else np.nan,
            "r2_ceiling": max(var_obs - noise_var, 0.0) / var_obs if var_obs else np.nan,
        })
    out = pd.DataFrame(rows).set_index("poss_floor")
    out.attrs["var_true_fit"] = var_true
    out.attrs["noise_scale_k"] = float(slope)
    return out


# --------------------------------------------------------------------------- #
# Models, baselines, metrics
# --------------------------------------------------------------------------- #
def build_ridge():
    """Standardize, then ridge with a CV-selected alpha (see LINEUP_ALPHAS)."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=LINEUP_ALPHAS))


def build_ols():
    """Standardize, then plain OLS.

    Standardizing changes nothing about OLS's predictions; it exists so the OLS
    and ridge coefficient vectors live on the same (standardized) scale and
    ||ridge|| / ||OLS|| is a meaningful measure of how much shrinkage happened.
    """
    return make_pipeline(StandardScaler(), LinearRegression())


def normalized_weights(w) -> np.ndarray:
    """Sample weights divided by their mean (see LINEUP_ALPHAS for why).

    Only the *scale* changes -- every relative weight, and therefore every
    possession-weighted metric in this file, is identical. What it fixes is
    ridge's alpha being measured against an arbitrarily inflated data term.
    """
    w = np.asarray(w, dtype=float)
    mean = w.mean()
    return w / mean if mean else w


def fit_models(X: pd.DataFrame, y: pd.Series, w: pd.Series) -> dict:
    """Fit OLS, ridge and gradient boosting on identical features/weights.

    The GBM is left at sklearn's defaults (plus a fixed random_state): tuning
    it here would need a validation season carved out of train, and picking
    hyperparameters after seeing the test seasons is exactly the "tune toward
    the expected winner" failure this comparison is meant to avoid.
    """
    w = normalized_weights(w)
    ols = build_ols().fit(X, y, linearregression__sample_weight=w)
    ridge = build_ridge().fit(X, y, ridgecv__sample_weight=w)
    gbm = GradientBoostingRegressor(**GBM_PARAMS).fit(X, y, sample_weight=w)
    return {"OLS": ols, "Ridge": ridge, "GBM": gbm}


def coef_series(model, feature_cols: list[str]) -> pd.Series:
    """Standardized coefficients of a (scaler, linear model) pipeline."""
    return pd.Series(model[-1].coef_, index=feature_cols)


def regularization_report(models: dict, feature_cols: list[str], w_raw) -> dict:
    """Did ridge actually regularize, and is its alpha inside the grid?

    `interior` is the check that matters: an alpha pinned at either end of the
    grid means the search was truncated, not converged -- which is exactly how
    the raw-weight bug hid (alpha pinned at train_model.py's 1000 maximum).
    """
    ridge = models["Ridge"]
    alpha = float(ridge[-1].alpha_)
    ridge_norm = float(np.linalg.norm(coef_series(ridge, feature_cols)))
    ols_norm = float(np.linalg.norm(coef_series(models["OLS"], feature_cols)))
    return {
        "alpha": alpha,
        "interior": bool(LINEUP_ALPHAS[0] < alpha < LINEUP_ALPHAS[-1]),
        "coef_norm_ratio": ridge_norm / ols_norm if ols_norm else np.nan,
        "ridge_coef_norm": ridge_norm,
        "ols_coef_norm": ols_norm,
        "mean_raw_weight": float(np.mean(w_raw)),
    }


def weighted_metrics(y, pred, w) -> dict:
    """Possession-weighted R2 / MAE / signed bias, in DEF_RATING points."""
    return {
        "R2": r2_score(y, pred, sample_weight=w),
        "MAE": mean_absolute_error(y, pred, sample_weight=w),
        "BIAS": float(np.average(np.asarray(pred) - np.asarray(y), weights=w)),
    }


def baseline_predictions(
    y_train, w_train, meta_train, meta_test, offset_train=None, offset_test=None
) -> dict[str, np.ndarray]:
    """The two baselines every model has to beat, as test-set predictions.

    * Constant: the possession-weighted train mean DEF_RATING. A model that
      can't beat this has no signal at all.
    * Naive box score: BLK + STL + DREB summed over the five players' prior
      per-game lines, fit as a single weighted-OLS feature (the raw sum isn't
      in DEF_RATING units, so it needs the one coefficient to be scorable).

    `offset_*` (a per-row season league average) makes the same two baselines
    available for the season-centered target: they are fit on the centered
    target and returned back in raw DEF_RATING points, so every row of the
    comparison table is scored against the same raw labels.
    """
    off_tr = np.zeros(len(meta_train)) if offset_train is None else np.asarray(offset_train, dtype=float)
    off_te = np.zeros(len(meta_test)) if offset_test is None else np.asarray(offset_test, dtype=float)
    y_c = np.asarray(y_train, dtype=float) - off_tr

    const = float(np.average(y_c, weights=w_train))
    naive_tr = meta_train[["NAIVE_BOX_SUM"]].to_numpy(dtype=float)
    naive_te = meta_test[["NAIVE_BOX_SUM"]].to_numpy(dtype=float)
    naive_fit = LinearRegression().fit(naive_tr, y_c, sample_weight=normalized_weights(w_train))
    return {
        CONST_BASELINE: const + off_te,
        NAIVE_BASELINE: naive_fit.predict(naive_te) + off_te,
    }


# --------------------------------------------------------------------------- #
# Rolling-origin (expanding-window) cross-validation
# --------------------------------------------------------------------------- #
# Each fold trains on every season strictly before its single test season, so
# the model is only ever asked to predict forward -- the same thing the program
# would actually do -- and every season from the (MIN_TRAIN_SEASONS + 1)-th
# onward contributes a held-out fold. This replaces the old fixed 2-season
# holdout (train_model.py's N_TEST_SEASONS convention), which spent 10 of 12
# seasons on training and then reported a single number from 159 lineups: with
# ~150-1000 lineups per test season and correlated rows inside each, one split
# cannot distinguish a real effect from one season's weather.
MIN_TRAIN_SEASONS = 6  # -> fold 1 trains on the first 6 seasons, tests the 7th
CI_LEVEL = 0.95


def rolling_origin_folds(seasons: list[str]) -> list[tuple[list[str], str]]:
    """[(train_seasons, test_season), ...], expanding, chronological, no shuffle."""
    seasons = sorted(seasons)
    return [(seasons[:i], seasons[i]) for i in range(MIN_TRAIN_SEASONS, len(seasons))]


def fit_fold(X_train, y_train, w_train, meta_train) -> dict:
    """Fit both target variants on one fold's training block.

    Returns {"raw": models, "centered": models} -- the same three estimators
    twice, once on DEF_RATING and once on DEF_RATING minus that season's league
    average (see centered_target_models()).
    """
    return {
        "raw": fit_models(X_train, y_train, w_train),
        "centered": centered_target_models(X_train, y_train, w_train, meta_train),
    }


def predict_fold(models_by_variant: dict, X_test, meta_test) -> dict[tuple[str, str], np.ndarray]:
    """Test-set predictions in raw DEF_RATING points, keyed (model, variant)."""
    out = {}
    for variant, models in models_by_variant.items():
        for name, model in models.items():
            out[(name, variant)] = (
                model.predict(X_test) if variant == "raw"
                else centered_predictions(model, X_test, meta_test)
            )
    return out


def cv_evaluate(X: pd.DataFrame, meta: pd.DataFrame, keep_cols: list[str] | None = None) -> dict:
    """Rolling-origin CV over seasons.

    Returns:
      * fold_metrics -- one row per (test season, model, variant) with
        possession-weighted R2/MAE/bias and the fold's sizes.
      * oof -- out-of-fold predictions for every held-out lineup, one column per
        (model, variant). These are what the reliability and team-change
        diagnostics run on, so those diagnostics see six held-out seasons
        instead of the two the old holdout left.
      * regs -- per-fold ridge alpha / shrinkage, so a pinned alpha can't hide.
    """
    cols = list(X.columns) if keep_cols is None else keep_cols
    Xc = X[cols]
    y, w = meta[TARGET], meta["POSS_TOTAL"]
    league = season_league_average()
    folds = rolling_origin_folds(sorted(meta["SEASON"].unique()))

    rows, regs = [], []
    oof = pd.DataFrame(index=meta.index, dtype=float)
    oof_fold = pd.Series(index=meta.index, dtype=object)

    for train_seasons, test_season in folds:
        tr = meta["SEASON"].isin(train_seasons).to_numpy()
        te = (meta["SEASON"] == test_season).to_numpy()
        if not te.any() or tr.sum() < 50:
            continue

        fitted = fit_fold(Xc[tr], y[tr], w[tr], meta[tr])
        preds = predict_fold(fitted, Xc[te], meta[te])

        offset_tr = meta.loc[tr, "SEASON"].map(league).to_numpy(dtype=float)
        offset_te = meta.loc[te, "SEASON"].map(league).to_numpy(dtype=float)
        for variant, off_tr, off_te in [("raw", None, None), ("centered", offset_tr, offset_te)]:
            for name, pred in baseline_predictions(
                y[tr], w[tr], meta[tr], meta[te], off_tr, off_te
            ).items():
                preds[(name, variant)] = pred

        for (name, variant), pred in preds.items():
            m = weighted_metrics(y[te], pred, w[te])
            rows.append({
                "test_season": test_season, "model": name, "variant": variant,
                "train_lineups": int(tr.sum()), "test_lineups": int(te.sum()),
                "R2": m["R2"], "MAE": m["MAE"], "BIAS": m["BIAS"],
            })
            col = f"{name}|{variant}"
            if col not in oof:
                oof[col] = np.nan
            oof.loc[te, col] = pred

        oof_fold.loc[te] = test_season
        reg = regularization_report(fitted["raw"], cols, w[tr])
        reg["test_season"] = test_season
        regs.append(reg)

    oof = oof.loc[oof_fold.notna()]
    oof["FOLD"] = oof_fold.loc[oof.index]
    return {
        "folds": folds,
        "fold_metrics": pd.DataFrame(rows),
        "oof": oof,
        "regs": pd.DataFrame(regs).set_index("test_season"),
        "features": cols,
        # The last fold's fits: trained on every season but the most recent, so
        # they are the most-trained models that were still evaluated out of
        # sample. Used for the coefficient and sensitivity tables, which need
        # one fitted model rather than an average over folds.
        "last_models": fitted,
    }


def _t_crit(df: int) -> float:
    """Two-sided 95% t critical value (scipy ships with scikit-learn)."""
    from scipy import stats
    return float(stats.t.ppf(0.5 + CI_LEVEL / 2.0, df))


def cv_summary(res: dict, variant: str = "centered") -> pd.DataFrame:
    """Per model: mean per-fold R2 with a 95% CI, mean MAE, and pooled-OOF R2.

    The CI is across the season folds (n = number of folds, so df is small and
    the interval is wide). That is deliberate: a row-level bootstrap would be
    anti-conservative here, because lineups within a team-season share four of
    five players and one player's prior-season line is reused by every lineup he
    appears in -- the independent unit is closer to a season than a row.
    """
    fm = res["fold_metrics"]
    fm = fm[fm["variant"] == variant]
    out = []
    for name, grp in fm.groupby("model", sort=False):
        r2 = grp["R2"].to_numpy(dtype=float)
        k = len(r2)
        half = _t_crit(k - 1) * r2.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
        out.append({
            "model": name, "folds": k,
            "R2_mean": r2.mean(), "R2_lo95": r2.mean() - half, "R2_hi95": r2.mean() + half,
            "R2_worst_fold": r2.min(), "R2_best_fold": r2.max(),
            "MAE_mean": grp["MAE"].mean(), "BIAS_mean": grp["BIAS"].mean(),
        })
    return pd.DataFrame(out).set_index("model")


def pooled_oof_metrics(res: dict, meta: pd.DataFrame, variant: str = "centered") -> pd.DataFrame:
    """Metrics over all held-out folds pooled into one set (every fold's rows).

    Complements cv_summary(): the pooled number is possession-weighted across
    every held-out season at once, so a big season doesn't count the same as a
    small one the way it does in a mean of per-fold R2s.
    """
    oof, idx = res["oof"], res["oof"].index
    y, w = meta.loc[idx, TARGET], meta.loc[idx, "POSS_TOTAL"]
    rows = {}
    for col in oof.columns:
        if col == "FOLD" or not col.endswith(f"|{variant}"):
            continue
        m = weighted_metrics(y, oof[col], w)
        rows[col.split("|")[0]] = {"R2_pooled": m["R2"], "MAE_pooled": m["MAE"], "BIAS_pooled": m["BIAS"]}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Required diagnostic 1: reliability / calibration at team level
# --------------------------------------------------------------------------- #
def reliability(pred_test: np.ndarray, meta_test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Roll held-out lineup predictions up to team level and calibrate them.

    For each held-out (SEASON, TEAM_ID), the team's estimated DEF_RATING is the
    possession-weighted mean of its qualifying lineups' predictions. Regressing
    the team's TRUE DEF_RATING (team_season_stats.csv) on that estimate gives
    the numbers needed to wire this into a margin-based win model:

        DRtg_used = intercept + slope * DRtg_estimated

    A slope below 1 says the estimate is over-dispersed and must be shrunk
    toward the mean by that factor; the R2 says how much of the between-team
    spread in real defense the estimate explains at all.

    Caveat the number carries: the estimate covers only lineups above the
    possession floor (roughly a team's most-used units), while the team's true
    DEF_RATING covers all of its minutes.
    """
    est = meta_test.assign(pred=pred_test)
    weighted = (
        est.assign(_wp=est["pred"] * est["POSS_TOTAL"])
        .groupby(["SEASON", "TEAM_ID"])
        .agg(pred_sum=("_wp", "sum"), poss=("POSS_TOTAL", "sum"), lineups=("pred", "size"))
    )
    weighted["DRTG_EST"] = weighted["pred_sum"] / weighted["poss"]

    teams = pd.read_csv(TEAM_CSV).set_index(["SEASON", "TEAM_ID"])[TARGET].rename("DRTG_TRUE")
    tbl = weighted.join(teams, how="inner").reset_index()

    slope, intercept = np.polyfit(tbl["DRTG_EST"], tbl["DRTG_TRUE"], 1)
    fitted = intercept + slope * tbl["DRTG_EST"]
    # The number the calibrated MAE has to beat: predict each season's
    # league-average team DEF_RATING for every team in it. Without this, a
    # small team-level MAE reads as accuracy when it may just be regression to
    # a mean that was already known.
    league_pred = tbl["SEASON"].map(season_league_average()).to_numpy(dtype=float)
    stats = {
        "n_team_seasons": len(tbl),
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(r2_score(tbl["DRTG_TRUE"], fitted)),
        "sd_estimate": float(tbl["DRTG_EST"].std()),
        "sd_true": float(tbl["DRTG_TRUE"].std()),
        "MAE_calibrated": float(mean_absolute_error(tbl["DRTG_TRUE"], fitted)),
        "MAE_uncalibrated": float(mean_absolute_error(tbl["DRTG_TRUE"], tbl["DRTG_EST"])),
        "MAE_league_average": float(mean_absolute_error(tbl["DRTG_TRUE"], league_pred)),
        "lineups_per_team_season": float(tbl["lineups"].mean()),
    }
    return tbl, stats


# --------------------------------------------------------------------------- #
# Required diagnostic 2: team-change (context-memorization) leakage test
# --------------------------------------------------------------------------- #
def team_change_test(pred_test: np.ndarray, meta_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Accuracy on held-out lineups that stayed together vs. lineups with movers.

    Group A: all five players were on this same team the prior season.
    Group B: at least one player was somewhere else.

    If the model were partly memorizing team context (this team's scheme
    defends well) rather than learning about players, group A would be much
    more accurate than group B. Comparable accuracy on movers is evidence the
    signal travels with the players. Caveat: player_all_seasons.csv stores only
    a traded player's final-team stint per season (documented in
    pull_player_seasons_all.py), so "prior team" is his last team of that
    season -- a mid-prior-season trade can label a player either way.
    """
    rows = []
    for label, sel in [
        ("A: all 5 same team last season", meta_test["ALL_SAME_TEAM"].to_numpy(dtype=bool)),
        ("B: >=1 player changed teams", ~meta_test["ALL_SAME_TEAM"].to_numpy(dtype=bool)),
    ]:
        if not sel.any():
            continue
        m = weighted_metrics(y_test[sel], np.asarray(pred_test)[sel], meta_test.loc[sel, "POSS_TOTAL"])
        rows.append({
            "group": label,
            "lineups": int(sel.sum()),
            "poss_share": float(meta_test.loc[sel, "POSS_TOTAL"].sum() / meta_test["POSS_TOTAL"].sum()),
            "test_R2": m["R2"],
            "test_MAE": m["MAE"],
            "test_bias": m["BIAS"],
        })
    return pd.DataFrame(rows).set_index("group")


# --------------------------------------------------------------------------- #
# Era drift: raw vs. season-centered target
# --------------------------------------------------------------------------- #
@cache
def season_league_average() -> pd.Series:
    """League-average team DEF_RATING per season (team_season_stats.csv).

    Cached: every centered fit/prediction needs it, and it's a fixed table.
    """
    teams = pd.read_csv(TEAM_CSV)
    return teams.groupby("SEASON")[TARGET].mean()


def centered_target_models(X_train, y_train, w_train, meta_train) -> dict:
    """Fit the same three models on the season-centered target.

    Their predictions are centered too -- add the target season's league average
    back (season_league_average()) before scoring or using them.
    """
    league = season_league_average()
    offset = meta_train["SEASON"].map(league).to_numpy(dtype=float)
    return fit_models(X_train, y_train - offset, w_train)


def centered_predictions(model, X, meta) -> np.ndarray:
    """Predictions from a season-centered model, back in raw DEF_RATING points."""
    league = season_league_average()
    return model.predict(X) + meta["SEASON"].map(league).to_numpy(dtype=float)


def group_intercept_features(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """X plus a per-group intercept dummy for the team-change diagnostic.

    DIAGNOSTIC ONLY -- never part of the model. It encodes current-season team
    identity (did these five end last season on the team they now play for),
    which a fictional roster has no analogue for; CONTINUITY_PAIR_FRAC is the
    roster-internal feature that IS usable. Its purpose is to separate a level
    effect from a slope effect: if giving each group its own intercept closes
    the MAE gap, the model was mis-levelling the two groups, not mis-ranking
    lineups inside them.
    """
    return X.assign(_ALL_SAME_TEAM=meta["ALL_SAME_TEAM"].astype(float))


# --------------------------------------------------------------------------- #
# Per-feature sensitivity (the perturbation test, adapted to this model)
# --------------------------------------------------------------------------- #
def perturbation_impact(model, X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Rank features by how much perturbing them moves the predicted DEF_RATING.

    Same diagnostic as perturbation_tests.perturbation_impact(), with two
    deliberate differences:
      * that version perturbs each feature by its own measured aggregation
        MAPE, because there the error being propagated is aggregation error.
        Here there is no aggregation step to measure, so each feature is moved
        by +/- 1 SD of its own held-out distribution -- a common scale, which
        makes the ranking a pure "how much does the fit lean on this column".
      * output is in DEF_RATING points, not wins (no x82 conversion).

    Same limitation as the original: one feature at a time, which assumes
    independent errors. That is false here by construction -- OREB_36_mean and
    OREB_PCT_mean, or the DREB_PCT max/min/std triple, move together -- which
    is exactly why the pruning below is judged by held-out R2 and not by this
    table alone.
    """
    X = X[feature_cols]
    baseline = model.predict(X)
    rows = []
    for feat in feature_cols:
        sd = float(X[feat].std())
        if not sd:
            rows.append({"feature": feat, "sd": 0.0, "mean_abs_drtg_impact": 0.0})
            continue
        X_plus, X_minus = X.copy(), X.copy()
        X_plus[feat] = X[feat] + sd
        X_minus[feat] = X[feat] - sd
        d_plus = model.predict(X_plus) - baseline
        d_minus = model.predict(X_minus) - baseline
        rows.append({
            "feature": feat, "sd": sd,
            "mean_abs_drtg_impact": float(np.mean((np.abs(d_plus) + np.abs(d_minus)) / 2.0)),
        })
    return pd.DataFrame(rows).set_index("feature").sort_values("mean_abs_drtg_impact", ascending=False)


# --------------------------------------------------------------------------- #
# Feature pruning (collinear blocks), evaluated under the same CV
# --------------------------------------------------------------------------- #
# Cumulative stages, each removing one collinearity the coefficient table
# exposed. Offensive columns other than the OREB pair are deliberately kept.
_OREB_DROP = ["OREB_36_mean", "OREB_PCT_mean"]
_PCT_PM_DROP = ["PCT_PLUSMINUS_mean"]
# DREB_PCT_max is the one kept: "the lineup's best defensive rebounder", the
# same one-elite-player rationale the order statistics exist for. Chosen on that
# rationale before seeing the result, not by trying all three and keeping the
# winner (which would be tuning on the evaluation).
_DREB_ORDER_DROP = ["DREB_PCT_std", "DREB_PCT_min"]
PRUNE_STAGES = [
    ("A: full", []),
    ("B: A minus OREB pair", _OREB_DROP),
    ("C: B minus PCT_PLUSMINUS", _OREB_DROP + _PCT_PM_DROP),
    ("D: C, DREB_PCT order -> max only", _OREB_DROP + _PCT_PM_DROP + _DREB_ORDER_DROP),
]
PRUNED_DROP_COLS = PRUNE_STAGES[-1][1]  # the production feature set


def kept_columns(all_cols: list[str], drop: list[str]) -> list[str]:
    return [c for c in all_cols if c not in drop]


# --------------------------------------------------------------------------- #
# Report helpers
# --------------------------------------------------------------------------- #
def fmt_ci(row) -> str:
    return f"{row['R2_mean']:+.3f} [{row['R2_lo95']:+.3f}, {row['R2_hi95']:+.3f}]"


def print_cv_report(res: dict, meta: pd.DataFrame, title: str, variant: str = "centered") -> pd.DataFrame:
    """Per-fold table + across-fold summary + pooled-OOF metrics for one run."""
    fm = res["fold_metrics"]
    print(f"\n--- {title}: per-fold {variant}-target metrics ---")
    pivot = fm[fm["variant"] == variant].pivot(index="test_season", columns="model", values="R2")
    sizes = fm[fm["variant"] == variant].groupby("test_season")[["train_lineups", "test_lineups"]].first()
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}", "display.width", 220):
        print(sizes.join(pivot).to_string())
    summary = cv_summary(res, variant)
    pooled = pooled_oof_metrics(res, meta, variant)
    out = summary.join(pooled)
    print(f"\n--- {title}: across {int(summary['folds'].iloc[0])} folds ({variant} target) ---")
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}", "display.width", 220):
        print(out[["R2_mean", "R2_lo95", "R2_hi95", "R2_worst_fold", "R2_best_fold",
                   "MAE_mean", "BIAS_mean", "R2_pooled", "MAE_pooled"]].to_string())
    return out


def print_reliability(label: str, pred, meta_rows: pd.DataFrame) -> dict:
    _, rel = reliability(np.asarray(pred), meta_rows)
    print(f"\n  --- {label} ({rel['n_team_seasons']} held-out team-seasons, "
          f"{rel['lineups_per_team_season']:.1f} qualifying lineups each) ---")
    print(f"  DRtg_used = {rel['intercept']:.3f} + {rel['slope']:.3f} x DRtg_estimated")
    print(f"  R2 = {rel['R2']:.3f}   sd(estimate) = {rel['sd_estimate']:.2f}   "
          f"sd(true) = {rel['sd_true']:.2f}")
    print(f"  team-level MAE: {rel['MAE_uncalibrated']:.2f} uncalibrated  ->  "
          f"{rel['MAE_calibrated']:.2f} calibrated   "
          f"(league-average baseline: {rel['MAE_league_average']:.2f})")
    return rel


def print_team_change(label: str, pred, meta_rows: pd.DataFrame, y_rows: pd.Series) -> pd.DataFrame:
    change = team_change_test(np.asarray(pred), meta_rows, y_rows)
    print(f"\n  --- {label} ---")
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        print(change.to_string())
    if len(change) == 2:
        print(f"  Gap (B minus A): MAE {change['test_MAE'].iloc[1] - change['test_MAE'].iloc[0]:+.3f} points, "
              f"bias spread {abs(change['test_bias'].iloc[0] - change['test_bias'].iloc[1]):.3f}, "
              f"R2 {change['test_R2'].iloc[0] - change['test_R2'].iloc[1]:+.3f}")
    return change


def oof_slice(res: dict, meta: pd.DataFrame, model: str, variant: str):
    """(predictions, meta rows, labels) for one model's out-of-fold predictions."""
    idx = res["oof"].index
    return res["oof"][f"{model}|{variant}"], meta.loc[idx], meta.loc[idx, TARGET]


def build_floor_dataset(poss_floor: float, players: pd.DataFrame, use_hustle: bool = False,
                        min_season: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    lineups = load_lineups(poss_floor, min_season=min_season)
    return build_lineup_features(lineups, players, use_hustle)


def print_dataset_summary(poss_floor: float, X: pd.DataFrame, meta: pd.DataFrame, summary: dict) -> None:
    s = summary
    print(f"\n{'=' * 78}\n=== Possession floor >= {poss_floor}: {len(X)} lineups, "
          f"{X.shape[1]} features, seasons {meta['SEASON'].min()}..{meta['SEASON'].max()} ===")
    print(f"Lineups considered: {s['lineups_considered']}  |  used: {s['lineups_used']}  |  "
          f"dropped -- no prior-season row (rookies etc.): {s['dropped_no_prior_season_row']}, "
          f"prior season too small (GP<{MIN_GP} or MIN<{MIN_MPG}): {s['dropped_prior_season_too_small']}, "
          f"no tracking row: {s['dropped_no_tracking_row']}, "
          f"missing feature value: {s['dropped_missing_feature_value']}")


def print_alphas(res: dict) -> None:
    regs = res["regs"]
    pinned = (~regs["interior"]).sum()
    print(f"\nRidge per fold: alpha {regs['alpha'].min():g}..{regs['alpha'].max():g} "
          f"(grid {LINEUP_ALPHAS[0]:g}..{LINEUP_ALPHAS[-1]:g}, "
          f"{'all interior' if pinned == 0 else f'{pinned} PINNED AT GRID EDGE'})  |  "
          f"||ridge coef||/||OLS coef|| {regs['coef_norm_ratio'].min():.3f}..{regs['coef_norm_ratio'].max():.3f}"
          f"  |  mean raw possession weight {regs['mean_raw_weight'].mean():.0f} (mean-normalized before fitting)")


def main() -> None:
    print("=== 5-man lineup DEF_RATING from players' PRIOR-season individual stats ===")
    print("Strict one-season lag; no on-court team context (DEF/OFF/NET_RATING,")
    print("PLUS_MINUS, on/off) from any season -- see module docstring.")
    print("Evaluation: rolling-origin (expanding-window) CV by season, never random.")
    print("Headline metric is the SEASON-CENTERED target: league-average DEF_RATING")
    print("rose ~10 points across this sample, which contaminates raw-target R2.\n")

    players = load_player_features()
    print(f"Player-season feature rows: {len(players)}  (usable: {int(players['_COMPLETE'].sum())})")

    # Correctness gate before any modeling: the feature vector must not depend
    # on the order the five players are listed in.
    test_permutation_invariance(load_lineups(PRIMARY_POSS_FLOOR), players)

    # --- What accuracy is even achievable per floor --------------------------
    noise = label_noise_decomposition()
    print(f"\n=== Label noise by possession floor "
          f"(season-centered DEF_RATING; noise scale k={noise.attrs['noise_scale_k']:.0f}) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(noise.to_string())
    print("r2_ceiling = the highest R2 any model could reach at that floor if it predicted\n"
          "each lineup's TRUE defensive quality exactly; the rest of the observed spread is\n"
          "sampling noise in the label itself. Read every R2 below against it.")

    # --- Floor sweep under rolling-origin CV --------------------------------
    floors = {}
    for floor in POSS_FLOORS:
        X, meta, summary = build_floor_dataset(floor, players)
        print_dataset_summary(floor, X, meta, summary)
        res = cv_evaluate(X, meta)
        folds = res["folds"]
        print(f"Folds: {len(folds)} -- " + "; ".join(
            f"train {f[0][0]}..{f[0][-1]} -> test {f[1]}" for f in folds[:2]
        ) + f"; ... -> test {folds[-1][1]}")
        print_alphas(res)
        table = print_cv_report(res, meta, f"floor {floor}")
        floors[floor] = {"X": X, "meta": meta, "summary": summary, "res": res, "table": table}

    print(f"\n{'=' * 78}\n=== Floor choice under rolling-origin CV (season-centered target) ===")
    rows = []
    for floor, d in floors.items():
        t = d["table"]
        best = t.drop(index=[CONST_BASELINE, NAIVE_BASELINE])["R2_mean"].idxmax()
        # The floor also has to be judged on what the estimate is FOR: rolled up
        # to team level, does it track real team defense? A low floor keeps far
        # more of a team's minutes in the estimate (more qualifying lineups per
        # team-season) even though each individual label is noisier, so the
        # lineup-level and team-level answers need not agree -- that trade is
        # the whole point of putting both in one table.
        pred, meta_rows, _ = oof_slice(d["res"], d["meta"], best, "centered")
        _, rel = reliability(pred, meta_rows)
        rows.append({
            "poss_floor": floor,
            "lineups": len(d["X"]),
            "oof_lineups": len(d["res"]["oof"]),
            "r2_ceiling": noise.loc[floor, "r2_ceiling"],
            "best_model": best,
            "R2_mean": t.loc[best, "R2_mean"],
            "R2_lo95": t.loc[best, "R2_lo95"],
            "R2_hi95": t.loc[best, "R2_hi95"],
            "share_of_ceiling": t.loc[best, "R2_mean"] / noise.loc[floor, "r2_ceiling"],
            "MAE_mean": t.loc[best, "MAE_mean"],
            "MAE_gain": t.loc[CONST_BASELINE, "MAE_mean"] - t.loc[best, "MAE_mean"],
            "folds_R2_pos": int((d["res"]["fold_metrics"].query(
                "variant == 'centered' and model == @best")["R2"] > 0).sum()),
            "team_slope": rel["slope"],
            "team_R2": rel["R2"],
            "team_seasons": rel["n_team_seasons"],
            "team_lineups_each": rel["lineups_per_team_season"],
            "team_MAE_cal": rel["MAE_calibrated"],
            "team_MAE_league": rel["MAE_league_average"],
        })
    floor_tbl = pd.DataFrame(rows).set_index("poss_floor")
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}", "display.width", 240,
                           "display.max_columns", None):
        print(floor_tbl.to_string())
    print("team_* columns: the same reliability diagnostic as STEP 5, run per floor on that\n"
          "floor's out-of-fold predictions -- the floor is being chosen on the number that\n"
          "would actually be used, not only on lineup-level R2. Read team_seasons alongside\n"
          "them: a high floor leaves some team-seasons with no qualifying lineup at all, so\n"
          "the calibration is fit on fewer (and self-selected, more stable) teams.")
    print(f"\nPrimary floor in use: {PRIMARY_POSS_FLOOR} "
          f"(see PRIMARY_POSS_FLOOR for the reasoning this table has to support).")

    # --- Primary floor: what the fit leans on -------------------------------
    d = floors[PRIMARY_POSS_FLOOR]
    X, meta, res = d["X"], d["meta"], d["res"]
    all_cols = list(X.columns)
    last = res["last_models"]
    ridge_c = coef_series(last["centered"]["Ridge"], all_cols).sort_values(key=np.abs, ascending=False)
    ols_c = coef_series(last["centered"]["OLS"], all_cols)
    print(f"\n{'=' * 78}\n=== Primary floor {PRIMARY_POSS_FLOOR}: standardized coefficients "
          f"(final fold's season-centered fit) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        print(pd.DataFrame({"ridge": ridge_c, "OLS": ols_c.reindex(ridge_c.index)}).head(14).to_string())

    # --- STEP 3: does the continuity feature fix the group bias split? ------
    print(f"\n{'=' * 78}\n=== Roster continuity (prior-season teammate pairs) ===")
    print("CONTINUITY_PAIR_FRAC = fraction of the 10 player pairs who ended last season on\n"
          "the same team. Prior-season membership only -- no current-season performance, no\n"
          "minutes-together -- so it is computable before a season starts and for a roster\n"
          "that has never played (a cross-era fictional five scores 0.0).")
    cont = X["CONTINUITY_PAIR_FRAC"]
    print(f"\nDistribution over {len(X)} lineups: mean {cont.mean():.3f}, "
          f"zero for {(cont == 0).mean() * 100:.1f}% of lineups, "
          f"1.0 (all five together) for {(cont == 1).mean() * 100:.1f}%")
    print(f"Ridge coefficient (season-centered, final fold): "
          f"{ridge_c.get('CONTINUITY_PAIR_FRAC', float('nan')):+.3f} DEF_RATING points per 1 SD")

    no_cont = kept_columns(all_cols, CONTINUITY_FEATURES)
    res_nc = cv_evaluate(X, meta, keep_cols=no_cont)
    tbl_nc = print_cv_report(res_nc, meta, f"floor {PRIMARY_POSS_FLOOR} WITHOUT continuity")
    print(f"\nWith continuity (from the sweep above) vs without, season-centered R2 mean [95% CI]:")
    for name in ["OLS", "Ridge", "GBM"]:
        print(f"  {name:6s} with: {fmt_ci(d['table'].loc[name])}   without: {fmt_ci(tbl_nc.loc[name])}")

    print("\n--- Team-change diagnostic, out-of-fold, before vs after the continuity feature ---")
    for label, r in [("WITHOUT continuity", res_nc), ("WITH continuity", res)]:
        pred, meta_rows, y_rows = oof_slice(r, meta, "Ridge", "centered")
        print_team_change(f"Ridge (season-centered), {label}", pred, meta_rows, y_rows)

    # Level effect vs slope effect: give each group its own intercept.
    res_gi = cv_evaluate(group_intercept_features(X, meta), meta)
    pred, meta_rows, y_rows = oof_slice(res_gi, meta, "Ridge", "centered")
    print("\n--- DIAGNOSTIC ONLY (uses current-season team identity; not a usable feature) ---")
    print_team_change("Ridge (season-centered) + per-group intercept", pred, meta_rows, y_rows)

    # --- STEP 4: sensitivity, then prune the collinear blocks ---------------
    print(f"\n{'=' * 78}\n=== Per-feature sensitivity (perturbation test, +/-1 SD, DEF_RATING points) ===")
    oof_idx = res["oof"].index
    impact = perturbation_impact(last["centered"]["Ridge"], X.loc[oof_idx], all_cols)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(impact.head(16).to_string())
    print(f"(bottom 5: {', '.join(impact.tail(5).index)})")

    print(f"\n{'=' * 78}\n=== Pruning collinear blocks (same folds, same weights) ===")
    prune_rows = []
    prune_res = {}
    for label, drop in PRUNE_STAGES:
        keep = kept_columns(all_cols, drop)
        r = res if not drop else cv_evaluate(X, meta, keep_cols=keep)
        prune_res[label] = r
        t = cv_summary(r, "centered").join(pooled_oof_metrics(r, meta, "centered"))
        for name in ["OLS", "Ridge"]:
            prune_rows.append({
                "stage": label, "features": len(keep), "model": name,
                "R2_mean": t.loc[name, "R2_mean"], "R2_lo95": t.loc[name, "R2_lo95"],
                "R2_hi95": t.loc[name, "R2_hi95"], "MAE_mean": t.loc[name, "MAE_mean"],
                "R2_pooled": t.loc[name, "R2_pooled"],
            })
    prune_tbl = pd.DataFrame(prune_rows).set_index(["model", "stage"]).sort_index()
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}", "display.width", 220):
        print(prune_tbl.to_string())
    base = prune_tbl.loc[("Ridge", PRUNE_STAGES[0][0]), "R2_mean"]
    for label, _ in PRUNE_STAGES[1:]:
        delta = prune_tbl.loc[("Ridge", label), "R2_mean"] - base
        print(f"  Ridge season-centered R2 change, {label} vs full: {delta:+.4f}")

    # --- STEP 5: reliability / calibration on the pruned model, OOF only ----
    final_label, final_drop = PRUNE_STAGES[-1]
    res_final = prune_res[final_label]
    print(f"\n{'=' * 78}\n=== Reliability / calibration at team level "
          f"(out-of-fold, {len(res_final['oof'])} held-out lineups) ===")
    print(f"Feature set: {final_label} ({len(kept_columns(all_cols, final_drop))} features). "
          f"Team estimate = possession-weighted mean of that team-season's\nqualifying lineup "
          f"predictions; regression is true team DEF_RATING on that estimate.")
    for variant in ["raw", "centered"]:
        for name in ["OLS", "Ridge"]:
            pred, meta_rows, _ = oof_slice(res_final, meta, name, variant)
            print_reliability(f"{name} ({variant} target)", pred, meta_rows)

    # --- Optional hustle block, on its own reduced sample -------------------
    print(f"\n{'=' * 78}\n=== Optional hustle-stat block (needs prior seasons from "
          f"{HUSTLE_START_SEASON}) ===")
    min_lineup_season = prior_season_inverse(HUSTLE_START_SEASON)
    players_h = load_player_features(use_hustle=True)
    X_h, meta_h, _ = build_floor_dataset(PRIMARY_POSS_FLOOR, players_h, use_hustle=True,
                                         min_season=min_lineup_season)
    X_n, meta_n, _ = build_floor_dataset(PRIMARY_POSS_FLOOR, players, use_hustle=False,
                                         min_season=min_lineup_season)
    print(f"Reduced sample (lineup seasons {min_lineup_season} onward): {len(X_h)} lineups, "
          f"{X_h.shape[1]} vs {X_n.shape[1]} features; held-out folds are the same seasons,\n"
          "so this separates 'hustle helps' from 'losing early training seasons hurts'.")
    for label, Xv, metav in [("without hustle", X_n, meta_n), ("with hustle", X_h, meta_h)]:
        r = cv_evaluate(Xv, metav, keep_cols=kept_columns(list(Xv.columns), PRUNED_DROP_COLS))
        t = cv_summary(r, "centered")
        print(f"  {label:15s} Ridge {fmt_ci(t.loc['Ridge'])}  MAE {t.loc['Ridge', 'MAE_mean']:.3f}   "
              f"OLS {fmt_ci(t.loc['OLS'])}  MAE {t.loc['OLS', 'MAE_mean']:.3f}")


def prior_season_inverse(season: str) -> str:
    """'2016-17' -> '2017-18' (the first lineup season whose PRIOR season is `season`)."""
    start = int(season.split("-")[0]) + 1
    return f"{start}-{str(start + 1)[-2:]}"


if __name__ == "__main__":
    main()
