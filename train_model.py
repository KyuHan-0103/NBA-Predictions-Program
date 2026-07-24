"""Ridge regression on team-season stats.

Target: W_PCT (regular-season win percentage -- games-played-invariant).
Features: every numeric team stat except identifiers and GP.
Split: by season -- the most recent seasons are held out for test, so the model
is evaluated on seasons it never trained on (chronological holdout, mirroring
real forecasting use). With 10 seasons, 2 held out ~= an 80/20 split.

Ridge (L2) is used instead of plain OLS because several features remain
collinear (e.g. EFG_PCT / shooting counts, PACE / POSS); the L2 penalty shrinks
those unstable coefficients together. Features are standardized first so the
penalty applies evenly across columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The exact feature shape the model consumes."""
    return [c for c in df.columns if c not in DROP_COLS + [TARGET]]


def build_model():
    """Standardize, then ridge with a CV-selected alpha."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


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

    feature_cols = feature_columns(df)
    X_train, y_train = df.loc[~is_test, feature_cols], df.loc[~is_test, TARGET]
    X_test, y_test = df.loc[is_test, feature_cols], df.loc[is_test, TARGET]

    model = build_model()
    model.fit(X_train, y_train)
    chosen_alpha = model.named_steps["ridgecv"].alpha_

    print(f"Rows: {len(df)}  Features: {len(feature_cols)}  Target: {TARGET}")
    print(f"Train seasons ({len(train_seasons)}): {train_seasons[0]}..{train_seasons[-1]}"
          f"  ({len(X_train)} rows)")
    print(f"Test  seasons ({len(test_seasons)}): {', '.join(test_seasons)}"
          f"  ({len(X_test)} rows)")
    print(f"Ridge alpha (RidgeCV): {chosen_alpha:g}\n")

    for name, Xs, ys in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        m = metrics(model, Xs, ys)
        print(f"{name:5s}  R2={m['R2']:.3f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

    # Standardized coefficients: effect of a 1-SD change in each feature on W_PCT.
    coefs = pd.Series(
        model.named_steps["ridgecv"].coef_, index=feature_cols
    ).sort_values(key=np.abs, ascending=False)

    print("\n=== Standardized ridge coefficients (|impact| desc) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(coefs.to_string())


if __name__ == "__main__":
    main()
