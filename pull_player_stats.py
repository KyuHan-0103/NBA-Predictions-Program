"""Pull player-level regular-season stats for the current season from nba_api.

Produces one row per player-season (current season only) by pulling both Base and
Advanced measure types from nba_api's LeagueDashPlayerStats endpoint and merging
them on PLAYER_ID + SEASON. This script only pulls and stores data -- no modeling.

Column policy (mirrors pull_team_stats.py):
  * Redundant -- shared/internal duplicates: the estimated (E_*) and sp_work_*
    twins of the ratings, PACE_PER40, FGM_PG/FGA_PG, NICKNAME, WNBA_FANTASY_PTS.
  * Redundant -- derivable rates: FG_PCT / FG3_PCT / FT_PCT (makes & attempts are
    kept, so the percentage is recoverable).
  * Leakage -- team outcomes attributed to the player, not individual production:
    W, L, W_PCT, PLUS_MINUS, and the on/off ratings OFF_RATING / DEF_RATING /
    NET_RATING.
  * Season-total counting stats are converted to per-game (divided by GP). Rates
    already invariant to games played (percentages, per-100 ratings, PACE, usage,
    PIE) are left untouched.
"""

from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

import matplotlib.pyplot as plt

OUTPUT_CSV = "player_season_stats.csv"
REQUEST_TIMEOUT = 60  # seconds
SLEEP_BETWEEN_CALLS = 1.0  # be polite to stats.nba.com

KEYS = ["PLAYER_ID", "SEASON"]

# --- Drop policy ---
# Redundant: internal/duplicate columns and estimated (E_*) / sp_work_* twins.
REDUNDANT_DROP = [
    "NICKNAME", "WNBA_FANTASY_PTS", "PACE_PER40", "FGM_PG", "FGA_PG",
    "E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING", "E_PACE", "E_TOV_PCT", "E_USG_PCT",
    "sp_work_OFF_RATING", "sp_work_DEF_RATING", "sp_work_NET_RATING", "sp_work_PACE",
]
# Redundant derivable rates (makes + attempts are kept).
DERIVED_PCT_DROP = ["FG_PCT", "FG3_PCT", "FT_PCT"]
# Leakage / team outcomes / on-off results, not individual production.
LEAKAGE_DROP = ["W", "L", "W_PCT", "PLUS_MINUS", "OFF_RATING", "DEF_RATING", "NET_RATING"]

# Season-total counting stats to divide by games played (GP) -> per-game.
PER_GAME_COLS = [
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA",
    "PF", "PFD", "PTS", "POSS", "NBA_FANTASY_PTS", "DD2", "TD3",
]


def latest_completed_season_start_year(today: pd.Timestamp) -> int:
    """Start year of the most recently completed regular season.

    NBA seasons run ~Oct->Apr. By July of a given year, the season that began
    the previous October has finished, so its start year is `year - 1`.
    From October onward, the season starting that year is underway.
    """
    return today.year if today.month >= 10 else today.year - 1


def current_season(today: pd.Timestamp) -> str:
    """Return the current season label like '2025-26'."""
    start = latest_completed_season_start_year(today)
    return f"{start}-{str(start + 1)[-2:]}"


def drop_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every *_RANK column."""
    return df.drop(columns=[c for c in df.columns if c.endswith("_RANK")])


def to_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Divide season-total counting stats by GP so they're comparable across roles."""
    cols = [c for c in PER_GAME_COLS if c in df.columns]
    df[cols] = df[cols].div(df["GP"], axis=0)
    return df


def fetch_measure(season: str, measure_type: str) -> pd.DataFrame:
    """Fetch one season of per-player regular-season stats for a measure type."""
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        measure_type_detailed_defense=measure_type,
        timeout=REQUEST_TIMEOUT,
    )
    df = resp.get_data_frames()[0]
    df.insert(0, "SEASON", season)
    return df


def main() -> None:
    season = current_season(pd.Timestamp.today())
    print(f"Pulling player stats for {season}")

    print("  fetching base ...", flush=True)
    base = drop_rank_columns(fetch_measure(season, "Base"))
    time.sleep(SLEEP_BETWEEN_CALLS)

    print("  fetching advanced ...", flush=True)
    adv = drop_rank_columns(fetch_measure(season, "Advanced"))

    # Keep only keys + advanced-only columns to avoid duplicating shared fields
    # (PLAYER_NAME, TEAM_ID, GP, MIN, ...).
    adv_only = [c for c in adv.columns if c in KEYS or c not in base.columns]
    adv = adv[adv_only]

    data = base.merge(adv, on=KEYS, how="inner", validate="one_to_one")

    # --- Drop redundant + leakage columns ---
    data = data.drop(
        columns=REDUNDANT_DROP + DERIVED_PCT_DROP + LEAKAGE_DROP, errors="ignore"
    )

    # --- Convert season-total counting stats to per-game ---
    data = to_per_game(data)

    data.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"Dimensions: {data.shape[0]} rows x {data.shape[1]} cols")
    print(f"Unique players: {data['PLAYER_ID'].nunique()}")

    print("\n=== Column names ===")
    print(list(data.columns))

    print("\n=== First 5 rows ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(data.head())

    print("\n=== GP Distribution ===")
    data['GP'].plot(kind='hist', bins=30)
    plt.show()

    print("\n=== MIN Distribution ===")
    data['MIN'].plot(kind='hist', bins=30)
    plt.show()

if __name__ == "__main__":
    main()
