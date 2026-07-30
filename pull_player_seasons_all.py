"""Pull one row per player-season (Base + Advanced) for every season since
Advanced stats exist, for use by predict_fictional_roster.py's player lookup.

Season floor: 1996-97. Confirmed empirically against stats.nba.com --
LeagueDashPlayerStats' Advanced measure type (POSS, PACE, OREB_PCT, DREB_PCT,
PIE, ...) returns real, non-zero data from 1996-97 onward and an EMPTY frame
(zero rows, not an error, not zeroed columns) for any earlier season. That's a
different failure mode than the tracking-stat floor in pull_player_history.py
(which silently returns zeroed columns pre-2013-14) -- there is no known
silent-zero trap for Advanced player data at this floor, but
predict_fictional_roster.py still defensively checks for one per player-season
before accepting it onto a roster.

Unlike pull_player_history.py (2013-14 onward, needed for tracking columns),
this script pulls the full Advanced-stats era because the fictional-roster
tool needs to find historic players by season, not just recent ones.

Traded-player caveat (same as pull_player_history.py): LeagueDashPlayerStats
does not return a combined ("TOT") row for a player traded mid-season -- a
player traded in January shows only their new team's partial-season stat line
for that SEASON; their time with the old team is silently dropped, not
blended in.

This script only pulls and stores data -- no modeling.
"""

from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from aggregate_team import ADDITIVE_COLS

OUTPUT_CSV = "player_all_seasons.csv"
START_SEASON = "1996-97"  # first season with non-empty Advanced player data
REQUEST_TIMEOUT = 60  # seconds
SLEEP_BETWEEN_CALLS = 1.0  # be polite to stats.nba.com

KEYS = ["PLAYER_ID", "SEASON"]
# Kept from Base: identity/eligibility fields + the additive box-score events
# aggregate_team.py sums directly (imported from there so both scripts agree
# on exactly which counting stats matter).
BASE_EXTRA = ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN"] + ADDITIVE_COLS
# Kept from Advanced: the columns aggregate_team.py's aggregation needs
# (POSS, OREB_PCT, DREB_PCT, PACE) plus PIE, which predict_fictional_roster.py
# uses as the single-number "how good was this season" ranking metric for a
# player's "Prime" season.
ADV_EXTRA = ["POSS", "OREB_PCT", "DREB_PCT", "PACE", "PIE"]
# Season-total counting stats that must be divided by GP to become per-game.
_PER_GAME = ADDITIVE_COLS + ["MIN", "POSS"]


def recent_seasons(start_season: str, latest_start_year: int) -> list[str]:
    """Return season labels like '1996-97', ..., newest last."""
    start_year = int(start_season.split("-")[0])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, latest_start_year + 1)]


def latest_completed_season_start_year(today: pd.Timestamp) -> int:
    """Start year of the most recently completed regular season.

    NBA seasons run ~Oct->Apr. By July of a given year, the season that began
    the previous October has finished, so its start year is `year - 1`.
    From October onward, the season starting that year is underway.
    """
    return today.year if today.month >= 10 else today.year - 1


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


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch + merge Base and Advanced for one season, convert totals to per-game."""
    base = fetch_measure(season, "Base")[KEYS + BASE_EXTRA]
    time.sleep(SLEEP_BETWEEN_CALLS)
    adv = fetch_measure(season, "Advanced")[KEYS + ADV_EXTRA]
    time.sleep(SLEEP_BETWEEN_CALLS)

    df = base.merge(adv, on=KEYS, how="inner", validate="one_to_one")
    df = df[df["GP"] > 0].copy()  # guard the per-game division below
    df[_PER_GAME] = df[_PER_GAME].div(df["GP"], axis=0)
    return df


def main() -> None:
    today = pd.Timestamp.today()
    latest = latest_completed_season_start_year(today)
    seasons = recent_seasons(START_SEASON, latest)

    print(f"Pulling {len(seasons)} seasons: {seasons[0]} ... {seasons[-1]}")

    frames: list[pd.DataFrame] = []
    for season in seasons:
        print(f"  fetching {season} ...", flush=True)
        frames.append(fetch_season(season))

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(data)} player-seasons x {data.shape[1]} cols to {OUTPUT_CSV}")
    print(f"Unique players: {data['PLAYER_ID'].nunique()}")

    print("\n=== Column names ===")
    print(list(data.columns))

    print("\n=== First 10 rows ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(data.head(10))


if __name__ == "__main__":
    main()
