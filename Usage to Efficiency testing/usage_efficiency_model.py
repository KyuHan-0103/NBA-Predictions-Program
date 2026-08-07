"""Model how shooting efficiency (TS%) responds to a usage-rate (USG%) change,
fit separately for on-ball vs. off-ball shooter types, from real player-seasons
where the usage change was plausibly *externally caused* rather than natural
development.

Why "situation-change" pairs, not just any usage change
---------------------------------------------------------
Most usage changes are a player developing into (or out of) a role over their
career -- that's not the signal this model wants. Instead this only keeps
consecutive-season pairs where the player was already an established veteran
(AGE >= EXPERIENCE_AGE_GATE) *and* something external plausibly moved their
role: they were traded/signed elsewhere, or a high-usage teammate arrived or
departed. Neither trigger requires injury-report or transaction data -- both
are derived purely from the season-over-season stats already pulled by
pull_player_history.py.

Why shooter-type buckets
-------------------------
A ball-dominant creator (Russell Westbrook-type) losing usage loses the shots
he creates for himself; an off-ball shooter (catch-and-shoot/cutter-type)
losing usage loses lower-value reps his gravity used to generate for others.
There's no reason to expect the same USG%->TS% relationship for both, so they
are bucketed by an `on_ball_index` built from tracking stats (touches, time of
possession, pull-up-shot share) before fitting. A 3-bucket split was tested
against 2 and didn't improve out-of-sample accuracy (LOO MAE was slightly
*worse*, and the most on-ball tercile's slope flipped sign on a non-significant
fit) -- removed rather than kept as a false-precision option.

A second question this script tests: instead of a hard bucket cutoff, does
letting the USG%->TS% *slope itself* vary continuously with `on_ball_index`
(via an interaction term in a single regression) beat the discrete-bucket
model? Both are fit and compared on the same leave-one-out metrics; see
`fit_continuous` / `evaluate_buckets` and the comparison printed in `main`.

Why the taper
--------------
Real situation-change pairs never show a usage swing bigger than roughly
+/-15 percentage points. A fictional stacked roster run through
aggregate_team.py's usage-conservation scaling can imply far larger swings.
Extrapolating a fitted line into that region is unjustified, so predictions
are clipped to the observed training range per bucket (see `predict_ts_delta`)
-- beyond that range the predicted effect holds flat rather than growing
without bound.

This script only builds and validates the model -- it is not wired into
aggregate_team.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import HuberRegressor, LinearRegression

INPUT_CSV = "player_history_stats.csv"

# --- Pair-mining eligibility ---
PAIR_MIN_GP = 50          # established rotation/starter role, not a cameo season
PAIR_MIN_MPG = 20
EXPERIENCE_AGE_GATE = 25  # season-Y age floor -- proxy for "not still developing"
USG_DELTA_GATE = 0.03     # 3 percentage points (USG_PCT is a 0-1 fraction)

# --- "High-usage teammate" definition, for the teammate-shift trigger ---
TEAMMATE_MIN_GP = 40
TEAMMATE_MIN_MPG = 20
TEAMMATE_MIN_USG = 0.20

# --- Bucketing reference population (broader than pair-mining eligibility, so
# the on/off-ball boundary isn't self-referential to the small pairs sample) ---
REF_MIN_GP = 40
REF_MIN_MPG = 15

MIN_PAIRS_PER_BUCKET = 15  # below this, report the fit as unreliable rather than hiding it

EXTRAPOLATION_TEST_DELTAS = [-0.20, -0.15, -0.10, 0.10, 0.15, 0.20]


def season_start_year(season: str) -> int:
    """'2013-14' -> 2013."""
    return int(season.split("-")[0])


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["SEASON_START_YEAR"] = df["SEASON"].map(season_start_year)
    return df


def add_on_ball_index(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Composite on-ball score: mean z-score of time-of-possession, seconds/touch,
    and pull-up share (share of catch-and-shoot + pull-up jumpers that are
    pull-ups). Z-scored against REF_MIN_GP/REF_MIN_MPG rows only, then applied
    to every row -- keeps the on/off-ball boundary tied to the broad player
    population, not just the small situation-change pairs subsample.
    """
    df = df.copy()
    jumper_total = df["CATCH_SHOOT_FGA"] + df["PULL_UP_FGA"]
    df["PULL_UP_SHARE"] = np.where(jumper_total > 0, df["PULL_UP_FGA"] / jumper_total, 0.5)

    ref = (df["GP"] >= REF_MIN_GP) & (df["MIN"] >= REF_MIN_MPG)
    components = ["TIME_OF_POSS", "AVG_SEC_PER_TOUCH", "PULL_UP_SHARE"]
    z_scores = []
    for c in components:
        mean, std = df.loc[ref, c].mean(), df.loc[ref, c].std()
        z_scores.append((df[c] - mean) / std)
    df["ON_BALL_INDEX"] = pd.concat(z_scores, axis=1).mean(axis=1)
    return df, df.loc[ref, "ON_BALL_INDEX"]


def high_usage_teammates(df: pd.DataFrame) -> dict[tuple[int, int], frozenset[int]]:
    """(TEAM_ID, SEASON_START_YEAR) -> set of PLAYER_IDs who were significant
    (high-minutes, high-usage) contributors on that team that season."""
    mask = (df["GP"] >= TEAMMATE_MIN_GP) & (df["MIN"] >= TEAMMATE_MIN_MPG) & (df["USG_PCT"] >= TEAMMATE_MIN_USG)
    sig = df.loc[mask, ["TEAM_ID", "SEASON_START_YEAR", "PLAYER_ID"]]
    out: dict[tuple[int, int], frozenset[int]] = {}
    for (team, year), grp in sig.groupby(["TEAM_ID", "SEASON_START_YEAR"]):
        out[(team, year)] = frozenset(grp["PLAYER_ID"])
    return out


def mine_pairs(df: pd.DataFrame, hu_teammates: dict[tuple[int, int], frozenset[int]]) -> pd.DataFrame:
    """Consecutive-season pairs where usage moved for a plausibly external reason."""
    eligible = df[(df["GP"] >= PAIR_MIN_GP) & (df["MIN"] >= PAIR_MIN_MPG)].copy()
    by_player_year = eligible.set_index(["PLAYER_ID", "SEASON_START_YEAR"])

    rows = []
    for (player_id, year), row_y in by_player_year.iterrows():
        key_y1 = (player_id, year + 1)
        if key_y1 not in by_player_year.index:
            continue
        row_y1 = by_player_year.loc[key_y1]
        # .loc on a non-unique-safe MultiIndex can return a DataFrame if duplicated;
        # PLAYER_ID+SEASON_START_YEAR is unique per the pull, so this is a Series.

        if row_y["AGE"] < EXPERIENCE_AGE_GATE:
            continue

        delta_usg = float(row_y1["USG_PCT"] - row_y["USG_PCT"])
        if abs(delta_usg) < USG_DELTA_GATE:
            continue

        team_y, team_y1 = row_y["TEAM_ID"], row_y1["TEAM_ID"]
        if team_y != team_y1:
            trigger = "traded"
        else:
            hu_y = hu_teammates.get((team_y, year), frozenset()) - {player_id}
            hu_y1 = hu_teammates.get((team_y1, year + 1), frozenset()) - {player_id}
            if hu_y != hu_y1:
                trigger = "teammate_shift"
            else:
                continue

        rows.append({
            "PLAYER_ID": player_id,
            "PLAYER_NAME": row_y["PLAYER_NAME"],
            "SEASON_Y": f"{year}-{str(year + 1)[-2:]}",
            "SEASON_Y1": f"{year + 1}-{str(year + 2)[-2:]}",
            "TRIGGER": trigger,
            "USG_Y": row_y["USG_PCT"],
            "USG_Y1": row_y1["USG_PCT"],
            "DELTA_USG": delta_usg,
            "TS_Y": row_y["TS_PCT"],
            "TS_Y1": row_y1["TS_PCT"],
            "DELTA_TS": float(row_y1["TS_PCT"] - row_y["TS_PCT"]),
            "ON_BALL_INDEX": row_y["ON_BALL_INDEX"],
        })
    return pd.DataFrame(rows)


def assign_buckets(pairs: pd.DataFrame, ref_on_ball_index: pd.Series) -> pd.Series:
    """Bucket pairs by ON_BALL_INDEX (season-Y role) via a median split of the
    broad reference population, not the pairs subsample itself."""
    cut = ref_on_ball_index.quantile(0.5)
    return pd.cut(pairs["ON_BALL_INDEX"], bins=[-np.inf, cut, np.inf], labels=["off_ball", "on_ball"])


def fit_bucket(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit one bucket's DELTA_TS ~ DELTA_USG: robust (Huber) production fit,
    plain OLS + Pearson r for transparency, and the taper boundary d_max."""
    x2 = x.reshape(-1, 1)
    huber = HuberRegressor().fit(x2, y)
    ols = LinearRegression().fit(x2, y)
    r, p = pearsonr(x, y) if len(x) >= 2 else (float("nan"), float("nan"))
    d_max = float(max(abs(x.min()), abs(x.max())))
    return {
        "n": len(x),
        "huber_slope": float(huber.coef_[0]),
        "huber_intercept": float(huber.intercept_),
        "ols_slope": float(ols.coef_[0]),
        "ols_intercept": float(ols.intercept_),
        "ols_r2": float(ols.score(x2, y)),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "d_max": d_max,
        "delta_usg_range": (float(x.min()), float(x.max())),
    }


def predict_ts_delta(delta_usg: float, fit: dict) -> float:
    """Clip-then-linear: never extrapolate the fitted slope beyond the observed
    training range -- the predicted effect holds flat past that boundary."""
    clipped = float(np.clip(delta_usg, -fit["d_max"], fit["d_max"]))
    return fit["huber_slope"] * clipped + fit["huber_intercept"]


def leave_one_out(x: np.ndarray, y: np.ndarray) -> dict:
    """LOO CV using the same clip-then-linear scheme, re-fit (incl. d_max) on
    each fold so the taper boundary is never derived from the held-out point."""
    n = len(x)
    if n < 5:
        return {"n": n, "mae": float("nan"), "sign_agreement": float("nan")}
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        fit_i = fit_bucket(x[mask], y[mask])
        preds[i] = predict_ts_delta(x[i], fit_i)
    mae = float(np.mean(np.abs(preds - y)))
    nonzero = y != 0
    sign_agreement = float(np.mean(np.sign(preds[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else float("nan")
    return {"n": n, "mae": mae, "sign_agreement": sign_agreement}


def evaluate_buckets(pairs: pd.DataFrame, ref_on_ball_index: pd.Series) -> dict:
    bucket_col = assign_buckets(pairs, ref_on_ball_index)
    fits, loos = {}, {}
    for bucket in bucket_col.cat.categories:
        sub = pairs[bucket_col == bucket]
        if len(sub) < 5:
            fits[bucket] = None
            loos[bucket] = {"n": len(sub), "mae": float("nan"), "sign_agreement": float("nan")}
            continue
        x, y = sub["DELTA_USG"].to_numpy(), sub["DELTA_TS"].to_numpy()
        fits[bucket] = fit_bucket(x, y)
        loos[bucket] = leave_one_out(x, y)
    valid = [v for v in loos.values() if not np.isnan(v["mae"])]
    n_valid = sum(v["n"] for v in valid)
    pooled_mae = sum(v["n"] * v["mae"] for v in valid) / max(n_valid, 1)
    pooled_sign_agreement = sum(v["n"] * v["sign_agreement"] for v in valid) / max(n_valid, 1)
    return {
        "bucket_col": bucket_col, "fits": fits, "loos": loos,
        "pooled_mae": pooled_mae, "pooled_sign_agreement": pooled_sign_agreement,
    }


def fit_continuous(pairs: pd.DataFrame) -> dict:
    """Fit DELTA_TS ~ DELTA_USG + ON_BALL_INDEX + DELTA_USG*ON_BALL_INDEX, so the
    effective USG%->TS% slope is a continuous function of on-ball-ness
    (`b_usg + b_inter * on_ball_index`) instead of a fixed per-bucket constant."""
    x_usg = pairs["DELTA_USG"].to_numpy()
    x_obi = pairs["ON_BALL_INDEX"].to_numpy()
    X = np.column_stack([x_usg, x_obi, x_usg * x_obi])
    y = pairs["DELTA_TS"].to_numpy()
    huber = HuberRegressor().fit(X, y)
    ols = LinearRegression().fit(X, y)
    d_max = float(max(abs(x_usg.min()), abs(x_usg.max())))
    return {
        "n": len(x_usg),
        "huber_intercept": float(huber.intercept_),
        "huber_b_usg": float(huber.coef_[0]),
        "huber_b_obi": float(huber.coef_[1]),
        "huber_b_inter": float(huber.coef_[2]),
        "ols_intercept": float(ols.intercept_),
        "ols_b_usg": float(ols.coef_[0]),
        "ols_b_obi": float(ols.coef_[1]),
        "ols_b_inter": float(ols.coef_[2]),
        "ols_r2": float(ols.score(X, y)),
        "d_max": d_max,
        "delta_usg_range": (float(x_usg.min()), float(x_usg.max())),
    }


def predict_ts_delta_continuous(delta_usg: float, on_ball_index: float, fit: dict) -> float:
    """Same clip-then-linear taper as predict_ts_delta, but the slope is
    evaluated at `on_ball_index` instead of looked up per-bucket."""
    clipped = float(np.clip(delta_usg, -fit["d_max"], fit["d_max"]))
    slope = fit["huber_b_usg"] + fit["huber_b_inter"] * on_ball_index
    return fit["huber_intercept"] + fit["huber_b_obi"] * on_ball_index + slope * clipped


def leave_one_out_continuous(pairs: pd.DataFrame) -> dict:
    """LOO CV for the continuous model, refitting all coefficients (incl. d_max)
    on each fold so the held-out point never leaks into its own prediction."""
    x_usg = pairs["DELTA_USG"].to_numpy()
    x_obi = pairs["ON_BALL_INDEX"].to_numpy()
    y = pairs["DELTA_TS"].to_numpy()
    n = len(pairs)
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        fit_i = fit_continuous(pairs.iloc[mask])
        preds[i] = predict_ts_delta_continuous(x_usg[i], x_obi[i], fit_i)
    mae = float(np.mean(np.abs(preds - y)))
    nonzero = y != 0
    sign_agreement = float(np.mean(np.sign(preds[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else float("nan")
    return {"n": n, "mae": mae, "sign_agreement": sign_agreement}


def main() -> None:
    df = load_data()
    df, ref_on_ball_index = add_on_ball_index(df)
    hu_teammates = high_usage_teammates(df)
    pairs = mine_pairs(df, hu_teammates)

    print(f"Player-seasons loaded: {len(df)}  |  situation-change pairs found: {len(pairs)}")
    print(f"  by trigger: {pairs['TRIGGER'].value_counts().to_dict()}")

    print("\n=== Qualifying pairs ===")
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", lambda v: f"{v:.3f}"):
        print(pairs[["PLAYER_NAME", "SEASON_Y", "SEASON_Y1", "TRIGGER", "DELTA_USG", "DELTA_TS"]]
              .sort_values("DELTA_USG").to_string(index=False))

    bucket_res = evaluate_buckets(pairs, ref_on_ball_index)
    print("\n=== 2-bucket model ===")
    for bucket in bucket_res["bucket_col"].cat.categories:
        fit, loo = bucket_res["fits"][bucket], bucket_res["loos"][bucket]
        if fit is None:
            print(f"  {bucket:10s}  n={loo['n']:3d}  -- too few pairs to fit reliably (< 5)")
            continue
        flag = "" if fit["n"] >= MIN_PAIRS_PER_BUCKET else "  [SMALL SAMPLE -- treat as directional only]"
        print(f"  {bucket:10s}  n={fit['n']:3d}{flag}")
        print(f"      huber: slope={fit['huber_slope']:+.4f}  intercept={fit['huber_intercept']:+.4f}")
        print(f"      ols:   slope={fit['ols_slope']:+.4f}  intercept={fit['ols_intercept']:+.4f}  "
              f"R2={fit['ols_r2']:.3f}  pearson_r={fit['pearson_r']:+.3f} (p={fit['pearson_p']:.3f})")
        print(f"      observed delta_usg range: [{fit['delta_usg_range'][0]:+.3f}, {fit['delta_usg_range'][1]:+.3f}]  "
              f"d_max={fit['d_max']:.3f}")
        print(f"      LOO: MAE={loo['mae']:.4f}  sign_agreement={loo['sign_agreement']:.2%}")
        sign_note = "matches expectation (usage down -> TS up)" if fit["huber_slope"] < 0 else "OPPOSITE of expectation"
        print(f"      sign check: slope is {'negative' if fit['huber_slope'] < 0 else 'positive'} -> {sign_note}")
    print(f"  pooled LOO: MAE={bucket_res['pooled_mae']:.4f}  sign_agreement={bucket_res['pooled_sign_agreement']:.2%}")

    cont_fit = fit_continuous(pairs)
    cont_loo = leave_one_out_continuous(pairs)
    print("\n=== Continuous model (slope varies with ON_BALL_INDEX) ===")
    print(f"  n={cont_fit['n']}")
    print(f"      huber: intercept={cont_fit['huber_intercept']:+.4f}  b_usg={cont_fit['huber_b_usg']:+.4f}  "
          f"b_obi={cont_fit['huber_b_obi']:+.4f}  b_inter={cont_fit['huber_b_inter']:+.4f}")
    print(f"      ols:   intercept={cont_fit['ols_intercept']:+.4f}  b_usg={cont_fit['ols_b_usg']:+.4f}  "
          f"b_obi={cont_fit['ols_b_obi']:+.4f}  b_inter={cont_fit['ols_b_inter']:+.4f}  R2={cont_fit['ols_r2']:.3f}")
    print(f"      implied slope at on_ball_index=-2 (very off-ball): {cont_fit['huber_b_usg'] - 2 * cont_fit['huber_b_inter']:+.4f}")
    print(f"      implied slope at on_ball_index=0  (average):        {cont_fit['huber_b_usg']:+.4f}")
    print(f"      implied slope at on_ball_index=+2 (very on-ball):   {cont_fit['huber_b_usg'] + 2 * cont_fit['huber_b_inter']:+.4f}")
    print(f"      observed delta_usg range: [{cont_fit['delta_usg_range'][0]:+.3f}, {cont_fit['delta_usg_range'][1]:+.3f}]  "
          f"d_max={cont_fit['d_max']:.3f}")
    print(f"      LOO: MAE={cont_loo['mae']:.4f}  sign_agreement={cont_loo['sign_agreement']:.2%}")

    print("\n=== Bucket model vs. continuous model ===")
    print(f"  2-bucket:   pooled LOO MAE={bucket_res['pooled_mae']:.4f}  sign_agreement={bucket_res['pooled_sign_agreement']:.2%}")
    print(f"  continuous: LOO MAE={cont_loo['mae']:.4f}  sign_agreement={cont_loo['sign_agreement']:.2%}")
    mae_improvement = (bucket_res["pooled_mae"] - cont_loo["mae"]) / bucket_res["pooled_mae"]
    print(f"  relative MAE improvement of continuous over buckets: {mae_improvement:+.1%}")
    winner = "continuous" if mae_improvement > 0.05 else "2-bucket"
    print(f"  -> using the {winner} model "
          f"({'continuous clearly beats buckets' if winner == 'continuous' else 'continuous does not clearly beat buckets -- keeping the simpler discrete model'})")

    print("\n=== Extrapolation / taper test (2-bucket model) ===")
    for bucket in bucket_res["bucket_col"].cat.categories:
        fit = bucket_res["fits"][bucket]
        if fit is None:
            continue
        print(f"  {bucket}: d_max={fit['d_max']:.3f}")
        for d in EXTRAPOLATION_TEST_DELTAS:
            unclipped = fit["huber_slope"] * d + fit["huber_intercept"]
            clipped = predict_ts_delta(d, fit)
            flag = "  <- taper engaged" if abs(d) > fit["d_max"] else ""
            print(f"      delta_usg={d:+.2f}   unclipped_pred={unclipped:+.4f}   tapered_pred={clipped:+.4f}{flag}")

    print("\n=== Extrapolation / taper test (continuous model) ===")
    for obi_label, obi in [("very off-ball (obi=-2)", -2.0), ("average (obi=0)", 0.0), ("very on-ball (obi=+2)", 2.0)]:
        print(f"  {obi_label}: d_max={cont_fit['d_max']:.3f}")
        for d in EXTRAPOLATION_TEST_DELTAS:
            slope = cont_fit["huber_b_usg"] + cont_fit["huber_b_inter"] * obi
            unclipped = cont_fit["huber_intercept"] + cont_fit["huber_b_obi"] * obi + slope * d
            clipped = predict_ts_delta_continuous(d, obi, cont_fit)
            flag = "  <- taper engaged" if abs(d) > cont_fit["d_max"] else ""
            print(f"      delta_usg={d:+.2f}   unclipped_pred={unclipped:+.4f}   tapered_pred={clipped:+.4f}{flag}")


if __name__ == "__main__":
    main()
