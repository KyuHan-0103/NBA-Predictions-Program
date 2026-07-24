"""Aggregate individual players into a synthetic team stat-line, and validate it.

Goal
----
Given 5-15 players, build a single team stat-line in the *exact* feature shape the
ridge model in train_model.py consumes (see TEAM_CSV header). The idea is to be able
to score hypothetical rosters with the team model.

Method
------
1. Eligibility filter: drop players with < MIN_GP games or < MIN_MPG minutes/game.
2. Each player is expressed as per-minute production; the roster's minutes are
   rescaled so they sum to TOTAL_TEAM_MINUTES (240 = 5 players x 48 minutes).
   For a real rotation whose minutes already sum to ~240 the scale is ~1, so the
   team line is essentially the sum of the players.
3. Feature families are combined differently:
     * ADDITIVE box-score events (PTS, REB, AST, ...): one event belongs to one
       player, so team = sum of players (after minute rescaling).
     * EXPOSURE stats (POSS): all five on-court players accrue them at once, so
       the player-sum is ~5x the team value -> divide by 5.
     * DERIVED rates (OFF_RATING, EFG_PCT, AST_TO, AST_RATIO, TM_TOV_PCT, PACE):
       recomputed from the aggregated box-score totals with standard formulas.
     * OPPONENT-DEPENDENT rates (DEF_RATING, OREB_PCT, DREB_PCT): can't be rebuilt
       from an offensive box score, so they're approximated from the players' own
       values. Two sub-cases: rebound *shares* (OREB_PCT/DREB_PCT) -- the five
       on-court players' individual rates sum to the team's, so they scale up by
       the ~5 players sharing the floor; DEF_RATING is a team-level rate each
       player experiences, so it's a minutes-weighted average. These are the
       softest estimates -- see the validation output.

Validation
----------
For each real team this season, take its qualifying players, aggregate them, and
compare the synthetic line to the team's actual line from team_season_stats.csv.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from train_model import build_model

TEAM_CSV = "team_season_stats.csv"
TEAM_DROP_COLS = ["SEASON", "TEAM_ID", "TEAM_NAME", "GP"]
TEAM_TARGET = "W_PCT"
GAMES_PER_SEASON = 82  # for expressing W_PCT as a win-loss record

MIN_GP = 5              # exclude players with fewer games
MIN_MPG = 6         # exclude players with fewer minutes per game
TOTAL_TEAM_MINUTES = 240.0  # 5 players * 48 minutes

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_CALLS = 1.0

# Box-score counting stats: each event is credited to one player -> team = sum.
ADDITIVE_COLS = [
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
    "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS",
]
# On-court exposure: shared by all 5 players on the floor -> player-sum is ~5x team.
EXPOSURE_COLS = ["POSS"]
# Rebound shares: the 5 on-court players' individual rates sum to the team's.
SHARE_COLS = ["OREB_PCT", "DREB_PCT"]
# Team-level rate each player experiences: minutes-weighted average.
AVERAGE_COLS = ["DEF_RATING"]
# Player columns needed from the Advanced measure type (per-game POSS + the rates).
ADV_KEEP = ["PLAYER_ID", "SEASON", "POSS", "DEF_RATING", "OREB_PCT", "DREB_PCT"]
# Counting stats that arrive as season totals and must be divided by GP.
_PER_GAME = ADDITIVE_COLS + ["MIN", "POSS"]


def _fetch(season: str, measure: str) -> pd.DataFrame:
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        measure_type_detailed_defense=measure,
        timeout=REQUEST_TIMEOUT,
    )
    df = resp.get_data_frames()[0]
    df.insert(0, "SEASON", season)
    return df


def load_players(season: str) -> pd.DataFrame:
    """Per-game player stats incl. DEF_RATING/rebound rates (not in the curated CSV)."""
    base = _fetch(season, "Base")
    time.sleep(SLEEP_BETWEEN_CALLS)
    adv = _fetch(season, "Advanced")
    df = base.merge(adv[ADV_KEEP], on=["PLAYER_ID", "SEASON"], validate="one_to_one")
    df[_PER_GAME] = df[_PER_GAME].div(df["GP"], axis=0)  # totals -> per game
    return df


def filter_players(players: pd.DataFrame) -> pd.DataFrame:
    """Drop players below the games / minutes-per-game thresholds."""
    keep = (players["GP"] >= MIN_GP) & (players["MIN"] >= MIN_MPG)
    return players.loc[keep].copy()


def aggregate_team(
    players: pd.DataFrame,
    feature_cols: list[str],
    total_minutes: float = TOTAL_TEAM_MINUTES,
) -> pd.Series:
    """Combine 5-15 players into one team stat-line in `feature_cols` order."""
    n = len(players)
    if not 5 <= n <= 15:
        print(f"  [warn] roster has {n} players (expected 5-15)")

    mpg = players["MIN"].to_numpy(dtype=float)
    total_mpg = mpg.sum()
    scale = total_minutes / total_mpg      # rescale roster minutes to 240
    weights = mpg / total_mpg              # minute share, sums to 1

    agg: dict[str, float] = {}

    # Additive box-score events (rescaled sum of per-game production).
    for c in ADDITIVE_COLS:
        agg[c] = scale * players[c].to_numpy(dtype=float).sum()

    # Exposure stats: divide the (rescaled) player-sum by the 5 on-court players.
    for c in EXPOSURE_COLS:
        agg[c] = scale * players[c].to_numpy(dtype=float).sum() / 5.0

    # Team minutes are game-minutes (~48), not the 240 player-minutes.
    agg["MIN"] = total_minutes / 5.0

    # Rates derived from the aggregated box-score totals.
    agg["EFG_PCT"] = (agg["FGM"] + 0.5 * agg["FG3M"]) / agg["FGA"]
    agg["AST_TO"] = agg["AST"] / agg["TOV"]
    plays = agg["FGA"] + 0.44 * agg["FTA"] + agg["AST"] + agg["TOV"]
    agg["AST_RATIO"] = 100.0 * agg["AST"] / plays
    poss_plays = agg["FGA"] + 0.44 * agg["FTA"] + agg["TOV"]
    agg["TM_TOV_PCT"] = agg["TOV"] / poss_plays  # stored as a fraction, not per-100
    agg["OFF_RATING"] = 100.0 * agg["PTS"] / agg["POSS"]
    agg["PACE"] = agg["POSS"]  # possessions per 48 minutes ~= per-game possessions

    # Opponent-dependent rates: approximated from the players' own values.
    scaled_min = weights * total_minutes  # per-player minutes, sums to 240
    team_minutes = total_minutes / 5.0    # 48 game-minutes
    # Rebound shares: the 5 on-court players' individual rates sum to the team's.
    for c in SHARE_COLS:
        agg[c] = float((players[c].to_numpy(dtype=float) * scaled_min).sum()) / team_minutes
    # Defensive rating: minutes-weighted average of what each player's unit allows.
    for c in AVERAGE_COLS:
        agg[c] = float((players[c].to_numpy(dtype=float) * weights).sum())

    missing = [c for c in feature_cols if c not in agg]
    if missing:
        raise ValueError(f"aggregation did not produce required features: {missing}")
    return pd.Series(agg)[feature_cols]


def record(wpct: float, games: int = GAMES_PER_SEASON) -> str:
    """Express a win percentage as a W-L record over `games`."""
    wins = int(round(min(max(wpct, 0.0), 1.0) * games))
    return f"{wins}-{games - wins}"


def compare(agg: pd.Series, real: pd.Series) -> pd.DataFrame:
    """Side-by-side aggregated vs. real with signed and percent error."""
    out = pd.DataFrame({"aggregated": agg, "real": real})
    out["diff"] = out["aggregated"] - out["real"]
    out["pct_err"] = 100.0 * out["diff"] / out["real"].replace(0, np.nan)
    return out


def main() -> None:
    teams_all = pd.read_csv(TEAM_CSV)
    feature_cols = [c for c in teams_all.columns if c not in TEAM_DROP_COLS + [TEAM_TARGET]]
    season = sorted(teams_all["SEASON"].unique())[-1]  # current season
    teams = teams_all[teams_all["SEASON"] == season].set_index("TEAM_ID")

    print(f"Season: {season}  |  team features: {len(feature_cols)}")
    print(f"Eligibility: GP >= {MIN_GP} and MIN >= {MIN_MPG} mpg\n")

    players = load_players(season)
    qualified = filter_players(players)
    print(f"Players: {len(players)} pulled -> {len(qualified)} qualify "
          f"({len(players) - len(qualified)} excluded)\n")

    # --- Detailed validation on one real team (the deepest qualifying rotation) ---
    # Take each team's top-15 by minutes so rosters honor the 5-15 contract.
    counts = qualified["TEAM_ID"].value_counts()
    focus_id = int(counts.index[0])
    roster = (
        qualified[qualified["TEAM_ID"] == focus_id]
        .sort_values("MIN", ascending=False)
        .head(15)
    )
    name = teams.loc[focus_id, "TEAM_NAME"]

    agg_line = aggregate_team(roster, feature_cols)
    real_line = teams.loc[focus_id, feature_cols].astype(float)

    print(f"=== Detailed validation: {name} ({len(roster)} players) ===")
    table = compare(agg_line, real_line)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(table.to_string())
    print(f"\nMean abs % error over features: {table['pct_err'].abs().mean():.2f}%")

    # --- Aggregate validation across all teams (mean abs % error per feature) ---
    errs = []
    agg_lines: dict[int, pd.Series] = {}
    for tid, grp in qualified.groupby("TEAM_ID"):
        grp = grp.sort_values("MIN", ascending=False).head(15)
        if tid not in teams.index or len(grp) < 5:
            continue
        a = aggregate_team(grp, feature_cols)
        agg_lines[tid] = a
        r = teams.loc[tid, feature_cols].astype(float)
        errs.append((100.0 * (a - r) / r.replace(0, np.nan)).abs())
    mape = pd.concat(errs, axis=1).mean(axis=1).sort_values()

    print(f"\n=== Validation across {len(errs)} teams: mean abs % error per feature ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(mape.to_string())
    print(f"\nOverall mean abs % error: {mape.mean():.2f}%")

    # --- Run both lines through the ridge model: predicted win records ---
    # Train on prior seasons only, so current-season predictions are out-of-sample.
    train_df = teams_all[teams_all["SEASON"] != season]
    model = build_model()
    model.fit(train_df[feature_cols], train_df[TEAM_TARGET])

    pred_real = float(model.predict(real_line.to_frame().T)[0])
    pred_agg = float(model.predict(agg_line.to_frame().T)[0])
    actual = float(teams.loc[focus_id, TEAM_TARGET])
    print(f"\n=== {name}: predicted win record (over {GAMES_PER_SEASON} games) ===")
    print(f"  Actual               : {record(actual):>6}  (W_PCT {actual:.3f})")
    print(f"  Model on real stats  : {record(pred_real):>6}  (W_PCT {pred_real:.3f})")
    print(f"  Model on aggregated  : {record(pred_agg):>6}  (W_PCT {pred_agg:.3f})")

    # Same comparison for every team.
    ids = list(agg_lines.keys())
    A = pd.DataFrame([agg_lines[i] for i in ids], index=ids)[feature_cols]
    R = teams.loc[ids, feature_cols].astype(float)
    wins = lambda wp: (np.clip(wp, 0, 1) * GAMES_PER_SEASON).round().astype(int)
    out = pd.DataFrame({
        "team": teams.loc[ids, "TEAM_NAME"].to_numpy(),
        "actual_W": wins(teams.loc[ids, TEAM_TARGET].to_numpy()),
        "pred_real_W": wins(model.predict(R)),
        "pred_agg_W": wins(model.predict(A)),
    })
    out["agg_err"] = out["pred_agg_W"] - out["actual_W"]
    out = out.sort_values("actual_W", ascending=False)

    print(f"\n=== Predicted win totals across {len(out)} teams (sorted by actual) ===")
    print(out.to_string(index=False))
    mae_real = (out["pred_real_W"] - out["actual_W"]).abs().mean()
    mae_agg = out["agg_err"].abs().mean()
    print(f"\nMAE in wins vs actual  ->  model on real stats: {mae_real:.1f}   "
          f"model on aggregated: {mae_agg:.1f}")


if __name__ == "__main__":
    main()
