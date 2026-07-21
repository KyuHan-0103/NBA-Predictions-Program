"""Pull team-level regular-season stats for all 30 NBA teams over the last 10 seasons.

Pulls both Base and Advanced measure types from nba_api's LeagueDashTeamStats
endpoint (one aggregated row per team per season) and merges them into a single
table keyed on TEAM_ID + SEASON. This script only pulls and stores data -- no modeling.

Column policy:
  * WIN_PCT (W_PCT) is kept -- games played per season isn't constant, so raw
    win totals aren't comparable across seasons.
  * Dropped as redundant: W, L, PLUS_MINUS, every *_RANK column, and NET_RATING
    from the advanced stats.
  * Season-total counting stats (points, rebounds, minutes, possessions, ...) are
    converted to per-game so they're comparable across seasons of differing length.
    Rates already invariant to games played (percentages, per-100 ratings, PACE,
    ratios, PIE) are left untouched.
"""

from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

OUTPUT_CSV = "team_season_stats.csv"
NUM_SEASONS = 10
REQUEST_TIMEOUT = 60  # seconds
SLEEP_BETWEEN_CALLS = 1.0  # be polite to stats.nba.com

KEYS = ["TEAM_ID", "SEASON"]
# Redundant raw columns to drop from the base stats (W_PCT is intentionally kept).
BASE_DROP = ["W", "L", "PLUS_MINUS", "FG3_PCT", "FT_PCT", "FG_PCT"]
# Redundant column to drop from the advanced stats.
ADVANCED_DROP = ["NET_RATING", "E_PACE", "E_OFF_RATING", "E_NET_RATING", "AST_PCT", "REB_PCT", "PACE_PER40", "TS_PCT", "E_DEF_RATING", "PIE"]

# Season-total counting stats that must be divided by games played (GP) to become
# per-game. Everything not listed here (W_PCT, *_PCT, *_RATING, PACE, PACE_PER40,
# AST_TO, AST_RATIO, PIE, ...) is already games-played-invariant and left as-is.
PER_GAME_COLS = [
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA",
    "PF", "PFD", "PTS", "POSS",
]


def recent_seasons(num_seasons: int, latest_start_year: int) -> list[str]:
    """Return season labels like '2025-26', newest last.

    `latest_start_year` is the calendar year the most recently completed
    regular season began (e.g. the 2025-26 season has start year 2025).
    """
    start_years = range(latest_start_year - num_seasons + 1, latest_start_year + 1)
    return [f"{y}-{str(y + 1)[-2:]}" for y in start_years]


def latest_completed_season_start_year(today: pd.Timestamp) -> int:
    """Start year of the most recently completed regular season.

    NBA seasons run ~Oct->Apr. By July of a given year, the season that began
    the previous October has finished, so its start year is `year - 1`.
    From October onward, the season starting that year is underway.
    """
    return today.year if today.month >= 10 else today.year - 1


def drop_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every *_RANK column."""
    return df.drop(columns=[c for c in df.columns if c.endswith("_RANK")])


def to_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Divide season-total counting stats by GP so they're comparable across seasons."""
    cols = [c for c in PER_GAME_COLS if c in df.columns]
    df[cols] = df[cols].div(df["GP"], axis=0)
    return df


def fetch_measure(season: str, measure_type: str) -> pd.DataFrame:
    """Fetch one season of per-team regular-season stats for a measure type."""
    resp = leaguedashteamstats.LeagueDashTeamStats(
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
    today = pd.Timestamp.today()
    latest = latest_completed_season_start_year(today)
    seasons = recent_seasons(NUM_SEASONS, latest)

    print(f"Pulling {len(seasons)} seasons: {seasons[0]} ... {seasons[-1]}")

    base_frames: list[pd.DataFrame] = []
    adv_frames: list[pd.DataFrame] = []
    for season in seasons:
        print(f"  fetching {season} (base) ...", flush=True)
        base_frames.append(fetch_measure(season, "Base"))
        time.sleep(SLEEP_BETWEEN_CALLS)
        print(f"  fetching {season} (advanced) ...", flush=True)
        adv_frames.append(fetch_measure(season, "Advanced"))
        time.sleep(SLEEP_BETWEEN_CALLS)

    # --- Base: drop ranks + redundant raw columns (keep W_PCT) ---
    base = drop_rank_columns(pd.concat(base_frames, ignore_index=True))
    base = base.drop(columns=BASE_DROP)

    # --- Advanced: drop ranks + NET_RATING, then keep only keys + columns that
    #     don't already exist in base (avoids duplicated GP/MIN/W_PCT/TEAM_NAME) ---
    adv = drop_rank_columns(pd.concat(adv_frames, ignore_index=True))
    # NET_RATING plus the same redundant raw columns dropped from base (the
    # advanced payload carries its own W/L, so drop them here too).
    adv = adv.drop(columns=ADVANCED_DROP + BASE_DROP, errors="ignore")
    adv_only = [c for c in adv.columns if c in KEYS or c not in base.columns]
    adv = adv[adv_only]

    # --- Merge on TEAM_ID + SEASON ---
    data = base.merge(adv, on=KEYS, how="inner", validate="one_to_one")

    # --- Convert season-total counting stats to per-game (GP not constant) ---
    data = to_per_game(data)

    data.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(data)} rows x {data.shape[1]} cols to {OUTPUT_CSV}")
    print(f"Teams per season: {data.groupby('SEASON')['TEAM_ID'].nunique().to_dict()}")

    print("\n=== Column names ===")
    print(list(data.columns))

    print("\n=== First 10 rows ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(data.head(10))


if __name__ == "__main__":
    main()
