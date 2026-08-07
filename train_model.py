"""Ridge regression on team-season stats.

Target: logit(W_PCT) -- the log-odds of a win, ln(p / (1-p)).
Features: every numeric team stat except identifiers and GP.
Split: by season -- the most recent seasons are held out for test, so the model
is evaluated on seasons it never trained on (chronological holdout, mirroring
real forecasting use). With 10 seasons, 2 held out ~= an 80/20 split.

Ridge (L2) is used instead of plain OLS because several features remain
collinear (e.g. EFG_PCT / shooting counts, PACE / POSS); the L2 penalty shrinks
those unstable coefficients together. Features are standardized first so the
penalty applies evenly across columns.

Why the logit target
--------------------
W_PCT is a proportion: it lives in [0, 1] and cannot leave it. A linear model
fit directly on it does not know that, and a ridge line has no upper or lower
stop -- push the features far enough and it will happily return a win rate of
2.44, which is what the five-star stress roster in aggregate_team.py actually
produced (a "200-win season" over 82 games). Fitting ln(p / (1-p)) instead and
inverting with the logistic 1 / (1 + e^-z) makes the bound structural: the
inverse cannot leave [0, 1] for any finite input, so no roster, however extreme,
can produce an impossible record. (Open interval in exact arithmetic; float64
rounds to exactly 1.0 past z ~= 36.7 -- see inv_logit.)

On real teams this changes almost nothing -- the logistic is near-linear across
the 0.12-0.89 band real teams occupy, so the holdout numbers move only slightly
(see README's Test log). The transform earns its place at the extremes, which
is exactly where the fictional-roster path operates.

Bounded is not supported
------------------------
A saturated prediction (W_PCT ~= 1.000) is the logistic running out of room, not
a forecast of an 82-0 season. Worse, the transform removes the old tell: an
impossible 2.44 announced itself, where a clean-looking 82-0 does not. So the
model ships with an extrapolation guard (fit_extrapolation_guard) that measures
how far a row sits from the real teams the model was fit on, with the "this is
extrapolation" threshold read off those real teams rather than assumed.
predict() and the guard are meant to be read together: the first is always in
range, the second says whether to believe it.

The guard's own headline finding is that every aggregated roster -- including a
real team's own starting five -- sits outside the real-team cloud, so its ratio
matters and its boolean does not. See fit_extrapolation_guard.

Intervals
---------
prediction_interval() / interval_report() attach a 90% interval to a predicted
win rate, built on the logit scale so it comes out correctly asymmetric near
the bounds, and reported as two separate components: the model's own held-out
error and the irreducible randomness of an 82-game season. Neither covers the
two terms that dominate a *fictional* roster's error -- the dropped DEF_RATING
and the extrapolation itself -- so interval_report names those instead of
folding them in at a guessed size.

All 27 team features are used, on the roster path as well: a smaller,
rates-only set was built and tested for the fictional path and rejected on
accuracy (see README's Test log). aggregate_team.coefficient_signs() prints
what that leaves standing -- 8 of 16 coefficients with a defensible expected
sign fit backwards, which is harmless on real teams and is a live caveat on an
extrapolated roster.
"""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "team_season_stats.csv"
TARGET = "W_PCT"
# Non-feature columns: identifiers + GP (used only to build per-game stats).
DROP_COLS = ["SEASON", "TEAM_ID", "TEAM_NAME", "GP"]
N_TEST_SEASONS = 2  # most-recent seasons held out for test
ALPHAS = np.logspace(-3, 3, 25)  # ridge penalty grid searched by RidgeCV
GAMES_PER_SEASON = 82  # for expressing a win rate as a win-loss record

# logit(0) and logit(1) are infinite. No real team-season is either (the pulled
# span is 0.122-0.890, i.e. 10-72 to 73-9), so this clip never binds on real
# data -- it's here so the transform is total rather than dependent on that
# staying true.
LOGIT_EPS = 1e-6
# Past this quantile of the training teams' own feature-space distance, a row is
# as unusual as the most unusual real teams -- see fit_extrapolation_guard.
EDGE_QUANTILE = 0.99
# Measured reference points for reading `distance / limit`, all from this repo's
# own data (see README's Test log for how each was produced):
#
#   real team-season the guard never saw ......... <= 1.13x  (4.4% exceed 1.0x)
#   real team's own 15-man rotation, aggregated .. ~1.7x  (max 2.4x)
#   real team's own top-5 rotation, aggregated ... ~1.9x  (max 3.3x)
#   five-star stress roster, conservation ON ..... ~5.1x
#   five-star stress roster, conservation OFF .... ~9.2x
#   five centres (rebound shares clipped) ........ see aggregate_team's own run
#
# The load-bearing line is the second and third: *every* roster put through
# aggregate_team() clears 1.0x, 100% of them, because the aggregation itself
# displaces a line off the real-team cloud before any fictional-ness is
# involved. So on the roster path the boolean flag carries no information and
# the ratio carries all of it. MARGINAL_RATIO is only the point past which a row
# has gone further than a real unseen team ever did.
MARGINAL_RATIO = 1.15
ROSTER_RATIO_SCALE = (
    "real unseen team <=1.1x | real 15-man rotation ~1.7x | "
    "real top-5 ~1.9x (max 3.3x) | five-star stress roster ~5.1x"
)

# --- Prediction intervals (see prediction_interval) ---------------------------
# Two-sided 90% -> 1.645 SDs each side of the point estimate.
INTERVAL_Z = 1.645
# What dropping DEF_RATING costs on the *real-team* chronological holdout, in
# wins of MAE -- the size of the error term the interval below cannot include,
# because a fictional roster has no DEF_RATING to measure the loss against.
# Measured on this repo's 360 team-seasons, 2 held out: 4.85 wins of MAE
# without DEF_RATING vs 2.53 with it, on the 27 features the model uses.
DEF_RATING_COST_WINS = 2.3


def logit(p):
    """Win rate -> log-odds. ln(p / (1-p)), clipped off the poles."""
    p = np.clip(np.asarray(p, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p))


def inv_logit(z):
    """Log-odds -> win rate. 1 / (1 + e^-z), never outside [0, 1].

    In exact arithmetic the interval is open, but float64 is not exact: above
    z ~= 36.7 the 1 swamps e^-z and this returns exactly 1.0. So a predicted
    W_PCT of 1.0 is reachable and a printed "82-0" is a real possible output --
    it means the logistic saturated, which is what the extrapolation guard is
    for. (The bottom end doesn't round: with the clip below, the smallest value
    returned is ~2e-22, not 0.0.)

    The clip on z only avoids an overflow warning inside exp. It sits far past
    where the result has already saturated, so it changes no returned value.
    """
    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The exact feature shape the model consumes."""
    return [c for c in df.columns if c not in DROP_COLS + [TARGET] + ["DEF_RATING"]]


def build_model():
    """Standardize, then ridge with a CV-selected alpha, fit on logit(W_PCT).

    TransformedTargetRegressor wires up the two halves described in the module
    docstring: fit() pushes y through logit() before the ridge ever sees it,
    and predict() pushes the ridge's output back through inv_logit(). Callers
    keep handing in and reading out plain W_PCT -- the only visible change is
    that predict() can no longer return a value outside (0, 1).
    """
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
        func=logit,
        inverse_func=inv_logit,
    )


def ridge_step(model):
    """The fitted RidgeCV inside a build_model() estimator.

    build_model() no longer returns a bare Pipeline, so `model.named_steps`
    doesn't reach the estimator any more. Anything wanting .coef_ or .alpha_
    (the coefficient table below, perturbation_tests.py) goes through here.
    """
    return model.regressor_.named_steps["ridgecv"]


def metrics(model, X, y) -> dict:
    """Scored on the W_PCT scale, not the logit scale -- predict() has already
    inverted the transform, so these stay comparable to the pre-logit numbers.
    """
    pred = model.predict(X)
    return {
        "R2": r2_score(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
    }


def logodds_contributions(model, row: pd.Series) -> pd.Series:
    """Each feature's own share of the predicted log-odds: z_score * coefficient.

    The model standardizes before the ridge, so a prediction decomposes exactly:
    `z = intercept + sum_j (standardized_j * coef_j)`. This returns that sum's
    terms, which is what turns "why did this roster predict 82-0" from a guess
    into a lookup -- the three largest terms name the features responsible.

    `row` must already be in the fitted feature order (aggregate_team's output
    is, by contract).
    """
    pipe = model.regressor_
    scaler = pipe.named_steps["standardscaler"]
    ridge = pipe.named_steps["ridgecv"]
    standardized = scaler.transform(row.to_frame().T)[0]
    return pd.Series(standardized * ridge.coef_, index=row.index)


def logit_residual_sd(model, X, y) -> float:
    """SD of the model's residuals **on the logit scale**, in log-odds.

    This is the spread the interval in prediction_interval() is built from, and
    it has to be measured where the model actually works -- in log-odds, the
    scale it fits -- rather than in W_PCT, where the same log-odds error is
    worth a very different number of wins at .500 than at .900.

    Measure it on held-out rows. Measured on training rows it is the ridge's
    own optimism, not its error.
    """
    resid = logit(np.asarray(y, dtype=float)) - logit(model.predict(X))
    return float(np.std(resid, ddof=1))


def holdout_logit_residual_sd(
    df: pd.DataFrame,
    feature_cols: list[str],
    fit=None,
    target: str = TARGET,
    n_test_seasons: int = N_TEST_SEASONS,
) -> float:
    """logit_residual_sd() over the same chronological holdout main() reports on.

    Callers that fit on *every* season (the fictional path does -- there is no
    real season to hold a fictional roster out of) still need an honest error
    estimate, so this refits on the earlier seasons only, measures there, and
    hands back the SD to attach to the all-seasons model's predictions.

    `fit(X, y) -> fitted model` defaults to this module's own build_model(); the
    fictional path passes its own fitter so the SD belongs to the model that
    will actually do the predicting.
    """
    fit = fit or (lambda X, y: build_model().fit(X, y))
    seasons = sorted(df["SEASON"].unique())
    is_test = df["SEASON"].isin(seasons[-n_test_seasons:])
    model = fit(df.loc[~is_test, feature_cols], df.loc[~is_test, target])
    return logit_residual_sd(model, df.loc[is_test, feature_cols], df.loc[is_test, target])


# --------------------------------------------------------------------------
# Extrapolation guard
# --------------------------------------------------------------------------


def fit_extrapolation_guard(X: pd.DataFrame, y=None) -> dict:
    """Record the region of feature space the real teams actually occupy.

    The ridge fit is evidence about rows that look like the team-seasons it was
    fit on, and nothing else. The logit transform keeps a prediction inside
    (0, 1) no matter how far outside that region a row sits, which removes the
    old tell -- a nonsense 2.44 win rate announced itself. This puts the tell
    back as an explicit measurement.

    Distance is Mahalanobis on the standardized features. Plain per-feature
    range checks are not enough on their own: a roster can sit inside every
    single feature's own min/max and still be a *combination* no team has ever
    produced (the mid-tier five-man roster used to check this lands 15.5 out
    while breaking only one feature's range). Mahalanobis is the version of
    distance that knows the correlation structure, which is where that shows
    up. The covariance is Ledoit-Wolf shrunk purely for conditioning -- PACE
    and POSS correlate at ~0.99, which leaves the raw covariance near-singular.

    Both thresholds are read off the training rows themselves rather than
    assumed from a chi-square table, because the assumption a chi-square
    threshold needs (multivariate normal features) isn't one this data has to
    honor:
      * ``edge``  -- the EDGE_QUANTILE (99th percentile) distance among the
        training teams. Between here and ``limit`` a row is as far out as the
        handful of strangest real teams: supported, but thinly.
      * ``limit`` -- the largest distance among the training teams. This is the
        point where extrapolation begins. Past it, a row is further from the
        league's center than *any* of the real team-seasons in the sample, so
        there is no observation anywhere in the training data speaking to it.

    What this turned out to say about the fictional path, which is worth knowing
    before reading a flag: *every* roster run through aggregate_team() lands past
    ``limit`` -- 100% of 180 real teams' own top-5 rotations (median 1.9x) and
    100% of their own top-15 rotations (median 1.7x), never mind a cross-era
    five. The aggregation step displaces a line off the real-team cloud all by
    itself, before any fictional-ness enters. So on the roster path the boolean
    is a constant and only the ratio is informative; see ROSTER_RATIO_SCALE for
    what its values mean. The three bands are still discriminating for real
    team-seasons, which is what train_model.py's own diagnostics use them for.

    `y` is optional; when given, the guard also remembers the win-rate span the
    real teams actually covered, so a prediction can be checked against it.
    """
    features = list(X.columns)
    scaler = StandardScaler().fit(X)
    cov = LedoitWolf().fit(scaler.transform(X))
    train_d = np.sqrt(cov.mahalanobis(scaler.transform(X)))

    guard = {
        "features": features,
        "scaler": scaler,
        "cov": cov,
        "n_train": int(len(X)),
        "train_distance": train_d,
        "edge": float(np.quantile(train_d, EDGE_QUANTILE)),
        "limit": float(train_d.max()),
        "feat_lo": X.min(),
        "feat_hi": X.max(),
        "target_lo": None if y is None else float(np.min(y)),
        "target_hi": None if y is None else float(np.max(y)),
    }
    return guard


def extrapolation_distance(guard: dict, X) -> np.ndarray:
    """Each row's Mahalanobis distance from the training teams' center."""
    X = pd.DataFrame(X)[guard["features"]]
    return np.sqrt(guard["cov"].mahalanobis(guard["scaler"].transform(X)))


def extrapolation_level(guard: dict, X) -> np.ndarray:
    """Per row: 'in_range' | 'edge' | 'extrapolation' (see the guard's docstring)."""
    d = extrapolation_distance(guard, X)
    return np.where(d > guard["limit"], "extrapolation",
                    np.where(d > guard["edge"], "edge", "in_range"))


def extrapolation_ratio(guard: dict, row: pd.Series) -> float:
    """One row's distance as a multiple of `limit` -- the number to quote.

    On the roster path the boolean flag is a constant (every aggregated roster
    clears 1.0x), so this ratio carries all of the information; see the guard's
    docstring and ROSTER_RATIO_SCALE.
    """
    return float(extrapolation_distance(guard, row.to_frame().T)[0]) / guard["limit"]


def out_of_range_features(guard: dict, row: pd.Series) -> pd.DataFrame:
    """The features of one row that fall outside every training team's range.

    The "why" behind a flag, and the part a user can act on: Mahalanobis gives
    one number, this says which columns produced it. A row can be flagged with
    this table empty -- that's the impossible-combination case above.
    """
    row = row.reindex(guard["features"]).astype(float)
    lo, hi = guard["feat_lo"], guard["feat_hi"]
    outside = (row < lo) | (row > hi)
    return pd.DataFrame({
        "value": row[outside],
        "real_min": lo[outside],
        "real_max": hi[outside],
    })


def extrapolation_report(guard: dict, row: pd.Series, pred: float | None = None) -> str:
    """A printable verdict on one stat-line, for the roster-scoring scripts."""
    d = float(extrapolation_distance(guard, row.to_frame().T)[0])
    level = str(extrapolation_level(guard, row.to_frame().T)[0])

    lines = [
        f"Extrapolation check: {level.upper().replace('_', ' ')}  "
        f"(feature-space distance {d:.1f}; {guard['n_train']} real team-seasons "
        f"span up to {guard['limit']:.1f}, 99th pct {guard['edge']:.1f})"
    ]
    if level == "extrapolation":
        ratio = d / guard["limit"]
        if ratio <= MARGINAL_RATIO:
            lines.append(
                f"  Only {ratio:.2f}x past the most unusual real team -- real teams do "
                f"step this far past a prior season's limit (4.2% of them, max 1.13x), "
                f"so read this as an unfamiliar shape, not an unsupported one."
            )
        else:
            lines.append(
                f"  {ratio:.1f}x the limit. Read the ratio, not the flag: every roster put "
                f"through the aggregation clears 1.0x, so the flag alone says nothing."
            )
            lines.append(f"  Measured scale -- {ROSTER_RATIO_SCALE}.")
    elif level == "edge":
        lines.append(
            "  Inside the training range, but only as far in as its strangest few "
            "teams -- thin support."
        )

    oor = out_of_range_features(guard, row)
    if not oor.empty:
        lines.append(f"  Outside every real team's own range ({len(oor)} features):")
        for feat, r in oor.iterrows():
            lines.append(
                f"    {feat:<11s} {r['value']:>9.3f}   real teams "
                f"{r['real_min']:.3f} .. {r['real_max']:.3f}"
            )

    if pred is not None and guard["target_lo"] is not None:
        if pred < guard["target_lo"] or pred > guard["target_hi"]:
            lines.append(
                f"  Predicted W_PCT {pred:.3f} is outside the real-team span "
                f"{guard['target_lo']:.3f} .. {guard['target_hi']:.3f} -- the logit kept it "
                f"in [0, 1], which is the most it promises. A saturated bound is not a "
                f"forecast."
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Prediction intervals
# --------------------------------------------------------------------------


def prediction_interval(
    pred: float,
    logit_sd: float,
    games: int = GAMES_PER_SEASON,
    z: float = INTERVAL_Z,
) -> dict:
    """A 90% interval around a predicted win rate, in two separate pieces.

    Built on the logit scale, then transformed back
    ------------------------------------------------
    The model fits log-odds, so the interval is `inv_logit(z_pred +- z * sd)`
    rather than `pred +- something`. That matters at the top of the range,
    which is exactly where the fictional path lives: a symmetric +-0.05 band
    around a predicted 0.95 runs past 1.0 and quotes an 82-win ceiling as if
    83 were available. Transforming an interval that is symmetric in log-odds
    comes back correctly asymmetric in wins, squeezed against the bound.

    Two components, reported separately
    -----------------------------------
    * ``model``   -- the model's own error, `logit_sd` (SD of its held-out
      residuals in log-odds, from logit_residual_sd). "How wrong is the model
      about this team's quality."
    * ``outcome`` -- irreducible: even a team whose true quality is known
      exactly does not post the same record twice. A season is `games`
      Bernoulli trials, so a realised win rate has SD `sqrt(p(1-p)/games)`.
      Computed on the rate scale, where it is exact and symmetric.
    * ``combined`` -- both at once. To add them they have to share a scale, so
      the outcome SD is carried to log-odds by the delta method: `d z / d p =
      1 / (p(1-p))`, giving `sqrt(p(1-p)/games) / (p(1-p)) =
      1 / sqrt(games * p * (1-p))`, and the two SDs add in quadrature.

    A consequence of that last step worth knowing: as `pred` approaches a bound
    the log-odds scale stretches without limit, so the combined interval gets
    *wider* than the outcome-only one even though the win counts are pinned.
    That is the transform being honest -- a saturated prediction carries no
    resolution -- not a defect to clamp away. At full saturation (`pred` at the
    LOGIT_EPS clip, i.e. inv_logit returned 1.0) it degenerates completely and
    the combined band covers 0-82; ``saturated`` in the result marks that case
    so a caller can say so rather than print a range that looks measured.

    Reference values (games=82, z=1.645, logit_sd=0.270), which this function
    reproduces exactly:

        pred     model     outcome   combined
        0.500    32-50     34-48     30-52
        0.700    49-64     51-64     46-66
        0.850    64-74     64-75     61-75
        0.950    76-79     75-81     72-80

    Returns W_PCT bounds and their win equivalents for each component.
    """
    p = float(np.clip(pred, LOGIT_EPS, 1.0 - LOGIT_EPS))
    z_pred = float(logit(p))
    var = p * (1.0 - p)

    sd_outcome_rate = np.sqrt(var / games)
    sd_outcome_logit = 1.0 / np.sqrt(games * var)
    sd_combined_logit = float(np.hypot(logit_sd, sd_outcome_logit))

    bands = {
        "model": (float(inv_logit(z_pred - z * logit_sd)),
                  float(inv_logit(z_pred + z * logit_sd))),
        # Rate scale, so this stays the exact binomial spread; clipped only
        # because p +- k*sd can leave [0, 1] near the ends.
        "outcome": (float(np.clip(p - z * sd_outcome_rate, 0.0, 1.0)),
                    float(np.clip(p + z * sd_outcome_rate, 0.0, 1.0))),
        "combined": (float(inv_logit(z_pred - z * sd_combined_logit)),
                     float(inv_logit(z_pred + z * sd_combined_logit))),
    }

    out = {
        "pred": p,
        "wins": int(round(p * games)),
        "games": games,
        "z": z,
        # inv_logit hit a bound and the clip in logit() is what is holding the
        # prediction finite -- the combined band below is meaningless there.
        "saturated": bool(p <= LOGIT_EPS or p >= 1.0 - LOGIT_EPS),
        "logit_sd": float(logit_sd),
        "sd_outcome_rate": float(sd_outcome_rate),
        "sd_outcome_logit": float(sd_outcome_logit),
        "sd_combined_logit": sd_combined_logit,
    }
    for name, (lo, hi) in bands.items():
        out[name] = (lo, hi)
        out[f"{name}_wins"] = (int(round(lo * games)), int(round(hi * games)))
    # The delta-method combination stops meaning anything before `saturated`
    # does: a few games short of the bound, `1 / sqrt(games * p * (1-p))` is
    # already large enough to open the band to the full range. Flag the symptom
    # (a band covering ~everything) rather than the cause, so a caller never
    # prints "0-82" as though it were a measurement.
    lo_w, hi_w = out["combined_wins"]
    out["degenerate"] = bool(hi_w - lo_w >= 0.9 * games)
    return out


def interval_report(
    pred: float,
    logit_sd: float,
    guard_ratio: float | None = None,
    games: int = GAMES_PER_SEASON,
    z: float = INTERVAL_Z,
    def_rating_cost_wins: float = DEF_RATING_COST_WINS,
    extra_uncovered: list[str] | None = None,
) -> str:
    """The printable interval, with an explicit list of what it leaves out.

    The omission notice is not boilerplate. `logit_sd` is measured on real
    teams the model was fit on the likes of; a fictional roster is neither
    (it sits outside the training region -- `guard_ratio` says how far) and is
    scored with DEF_RATING dropped, worth `def_rating_cost_wins` of MAE on real
    teams. Neither term is quantifiable from anything this project has, so they
    are named and left unquantified rather than folded in at a guessed size.
    An interval that silently omitted them would be the worst of the options:
    narrow, precise-looking, and missing its largest term.
    """
    iv = prediction_interval(pred, logit_sd, games=games, z=z)
    # Coverage implied by z, so the printed "90%" tracks INTERVAL_Z if it moves.
    pct = int(round(100 * (2 * 0.5 * (1 + math.erf(z / math.sqrt(2))) - 1)))

    def band(name: str) -> str:
        lo, hi = iv[f"{name}_wins"]
        return f"{lo}-{hi}"

    headline = ("combined band DEGENERATE -- see below" if iv["degenerate"]
                else f"{pct}% interval {band('combined')}")
    lines = [
        f"Predicted record: {iv['wins']}-{games - iv['wins']}   ({headline})",
        f"  Model error only ({logit_sd:.3f} log-odds SD on held-out real teams): "
        f"{band('model')}",
        f"  Outcome randomness only ({games} games at p={iv['pred']:.3f}): "
        f"{band('outcome')}",
        f"  Both combined: {band('combined')}",
    ]
    if iv["degenerate"]:
        cause = ("the logistic saturated -- inv_logit returned "
                 f"{1.0 if iv['pred'] > 0.5 else 0.0:.1f}"
                 if iv["saturated"] else
                 f"the prediction sits {min(iv['wins'], games - iv['wins'])} games from "
                 f"the bound")
        lines += textwrap.wrap(
            f"That combined band is not a measurement: {cause}, where log-odds "
            f"stretches without limit, so carrying the outcome SD onto that scale "
            f"opens the band to the whole range. The model-error band above is the "
            f"informative one; read the point estimate as 'past what this model can "
            f"express', not as a forecast.",
            width=74, initial_indent="  ", subsequent_indent="  ",
        )
    lines += [
        "  Interval covers: model error on held-out real teams, and game-to-game",
        f"  randomness over {games} games.",
    ]

    uncovered = [f"the dropped DEF_RATING (worth ~{def_rating_cost_wins:.1f} wins of "
                 f"accuracy on real teams)"]
    if guard_ratio is not None:
        uncovered.append(f"this roster's {guard_ratio:.1f}x extrapolation ratio")
    uncovered += extra_uncovered or []
    closer = "Neither" if len(uncovered) == 2 else "None of these"
    lines += textwrap.wrap(
        f"Interval does NOT cover: {' or '.join(uncovered)}. "
        f"{closer} is quantifiable from available data.",
        width=74, initial_indent="  ", subsequent_indent="  ",
    )
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    # --- Season-based split: hold out the most recent N seasons ---
    # Season labels like "2016-17" sort chronologically as plain strings.
    seasons = sorted(df["SEASON"].unique())
    test_seasons = seasons[-N_TEST_SEASONS:]
    train_seasons = seasons[:-N_TEST_SEASONS]
    is_test = df["SEASON"].isin(test_seasons)

    feature_cols = feature_columns(df)
    X_train, y_train = df.loc[~is_test, feature_cols], df.loc[~is_test, TARGET]
    X_test, y_test = df.loc[is_test, feature_cols], df.loc[is_test, TARGET]

    model = build_model()
    model.fit(X_train, y_train)
    chosen_alpha = ridge_step(model).alpha_

    print(f"Rows: {len(df)}  Features: {len(feature_cols)}  Target: logit({TARGET})")
    print(f"Train seasons ({len(train_seasons)}): {train_seasons[0]}..{train_seasons[-1]}"
          f"  ({len(X_train)} rows)")
    print(f"Test  seasons ({len(test_seasons)}): {', '.join(test_seasons)}"
          f"  ({len(X_test)} rows)")
    print(f"Ridge alpha (RidgeCV): {chosen_alpha:g}\n")

    # Metrics are on the W_PCT scale (predict() inverts the logit), so they stay
    # directly comparable to the pre-logit model's numbers.
    for name, Xs, ys in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        m = metrics(model, Xs, ys)
        print(f"{name:5s}  R2={m['R2']:.3f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}"
              f"  ({m['MAE'] * GAMES_PER_SEASON:.2f} wins)")

    # Standardized coefficients: effect of a 1-SD change in each feature on the
    # LOG-ODDS of a win, not on W_PCT directly -- that's what the model now fits.
    # The logistic's slope is steepest at .500 and flattens toward either end, so
    # a coefficient has no single win equivalent; wins_at_500 converts it at the
    # steepest point (d W_PCT / d z = p(1-p) = 0.25 at p = 0.5), which is the
    # largest win swing that coefficient can produce.
    coefs = pd.Series(
        ridge_step(model).coef_, index=feature_cols
    ).sort_values(key=np.abs, ascending=False)
    coef_tbl = pd.DataFrame({
        "logit_coef": coefs,
        "wins_at_500": coefs * 0.25 * GAMES_PER_SEASON,
    })

    print("\n=== Standardized ridge coefficients, log-odds per 1 SD (|impact| desc) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(coef_tbl.to_string())

    # --- Extrapolation guard, calibrated on every real team-season we have ---
    guard = fit_extrapolation_guard(df[feature_cols], df[TARGET])
    d = guard["train_distance"]
    print(f"\n=== Extrapolation guard (fitted on all {guard['n_train']} real team-seasons) ===")
    print(f"Feature-space distance over the real teams: median {np.median(d):.2f}  "
          f"p90 {np.quantile(d, 0.90):.2f}  p99 {guard['edge']:.2f}  max {guard['limit']:.2f}")
    print(f"  'edge' band begins at  {guard['edge']:.2f}  (p{EDGE_QUANTILE * 100:g} of real teams)")
    print(f"  EXTRAPOLATION begins at {guard['limit']:.2f}  (past every real team in the sample)")
    print(f"Real-team W_PCT span: {guard['target_lo']:.3f} .. {guard['target_hi']:.3f}  "
          f"({guard['target_lo'] * GAMES_PER_SEASON:.0f}-{GAMES_PER_SEASON - guard['target_lo'] * GAMES_PER_SEASON:.0f}"
          f" .. {guard['target_hi'] * GAMES_PER_SEASON:.0f}-{GAMES_PER_SEASON - guard['target_hi'] * GAMES_PER_SEASON:.0f}"
          f" over {GAMES_PER_SEASON} games)")

    # Sanity check on the threshold itself, using a guard fit on the TRAIN
    # seasons only -- the guard above includes the test rows, so scoring them
    # against it would be circular. The held-out teams are real teams, so they
    # should land almost entirely in range; a test season flagged as
    # extrapolation would say the league itself moved, not that a roster is
    # exotic, and would mean the threshold is drawn too tight.
    train_guard = fit_extrapolation_guard(X_train, y_train)
    held_out = pd.Series(extrapolation_level(train_guard, X_test)).value_counts()
    print(f"Held-out test rows scored against a train-only guard "
          f"(edge {train_guard['edge']:.2f}, limit {train_guard['limit']:.2f}): "
          f"{held_out.to_dict()}")


if __name__ == "__main__":
    main()
