"""Ridge vs. gradient boosting on team-season stats.

Target: W_PCT (regular-season win percentage -- games-played-invariant).
Features: every numeric team stat except identifiers and GP.
Split: by season -- the most recent seasons are held out for test, so both
models are evaluated on seasons they never trained on (chronological holdout,
mirroring real forecasting use). With 10 seasons, 2 held out ~= an 80/20 split.

Two models are trained on the identical split and compared:
  * Ridge (RidgeCV)  -- linear, L2-penalized; standardized features so the
    penalty applies evenly and collinear columns (PACE/POSS, shooting rates)
    shrink together instead of blowing up.
  * GradientBoosting -- additive shallow trees; captures nonlinearity and
    interactions, no scaling needed. Kept shallow (depth 2) given only ~240
    training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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
RANDOM_STATE = 42


def build_models() -> dict:
    """Return the models to compare, each a fit-able estimator/pipeline."""
    return {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.9,
            random_state=RANDOM_STATE,
        ),
    }


def metrics(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "R2": r2_score(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
    }


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    # --- Season-based split: hold out the most recent N seasons ---
    # Season labels like "2016-17" sort chronologically as plain strings.
    seasons = sorted(df["SEASON"].unique())
    test_seasons = seasons[-N_TEST_SEASONS:]
    train_seasons = seasons[:-N_TEST_SEASONS]
    is_test = df["SEASON"].isin(test_seasons)

    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
    X_train, y_train = df.loc[~is_test, feature_cols], df.loc[~is_test, TARGET]
    X_test, y_test = df.loc[is_test, feature_cols], df.loc[is_test, TARGET]

    print(f"Rows: {len(df)}  Features: {len(feature_cols)}  Target: {TARGET}")
    print(f"Train seasons ({len(train_seasons)}): {train_seasons[0]}..{train_seasons[-1]}"
          f"  ({len(X_train)} rows)")
    print(f"Test  seasons ({len(test_seasons)}): {', '.join(test_seasons)}"
          f"  ({len(X_test)} rows)\n")

    models = build_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = {
            "train": metrics(model, X_train, y_train),
            "test": metrics(model, X_test, y_test),
        }

    # --- Head-to-head comparison table ---
    print("=== Model comparison ===")
    header = f"{'Model':<18}{'Split':<7}{'R2':>8}{'RMSE':>9}{'MAE':>9}"
    print(header)
    print("-" * len(header))
    for name in models:
        for split in ("train", "test"):
            m = results[name][split]
            print(f"{name:<18}{split:<7}{m['R2']:>8.3f}{m['RMSE']:>9.4f}{m['MAE']:>9.4f}")

    ridge_test = results["Ridge"]["test"]
    gb_test = results["GradientBoosting"]["test"]
    winner = "Ridge" if ridge_test["RMSE"] <= gb_test["RMSE"] else "GradientBoosting"
    print(f"\nBest test RMSE: {winner} "
          f"(Ridge {ridge_test['RMSE']:.4f} vs GB {gb_test['RMSE']:.4f})")

    # --- Per-model explanation of drivers ---
    ridge = models["Ridge"].named_steps["ridgecv"]
    ridge_coefs = pd.Series(ridge.coef_, index=feature_cols).sort_values(
        key=np.abs, ascending=False
    )
    print(f"\n=== Ridge: top standardized coefficients (alpha={ridge.alpha_:g}) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(ridge_coefs.head(10).to_string())

    gb_imp = pd.Series(
        models["GradientBoosting"].feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\n=== GradientBoosting: top feature importances ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(gb_imp.head(10).to_string())


if __name__ == "__main__":
    main()
