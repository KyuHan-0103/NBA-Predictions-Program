"""Pull multi-season player Advanced + tracking stats for the usage/efficiency model.

Produces one row per player-season (2013-14 onward) with USG_PCT, TS_PCT, AGE,
TEAM_ID, and shot-location/touch tracking stats (catch-and-shoot, pull-up,
touches/time-of-possession). This script only pulls and stores data -- no
modeling (see usage_efficiency_model.py for that).

Season floor: 2013-14. SportVU/Second Spectrum player-tracking data does not
exist before that season -- confirmed empirically, the API doesn't error for
earlier seasons, it silently returns every tracking column as 0.0. Pulling
earlier seasons would look like real "nobody took catch-and-shoot jumpers"
data instead of "not tracked yet," so 2013-14 is a hard floor, not a style
choice.

Traded-player caveat: stats.nba.com's LeagueDashPlayerStats does not return a
combined ("TOT") row for players traded mid-season the way the .com UI does --
querying it here returns exactly one row per player per season, reflecting
only their *most recent* team stint that season (confirmed: no duplicate
PLAYER_IDs, no TEAM_ABBREVIATION == "TOT" rows). A player traded in, say,
January will show a partial-season GP/MIN/USG_PCT/TS_PCT for their new team
only -- their time with the old team is silently dropped, not blended in.
For this project's purposes that's actually fine: usage_efficiency_model.py
wants each season's row to reflect a stable *role*, and the post-trade stint
already isolates the role in the new situation. It also means a same-season
trade doesn't need special handling for cross-season pair-mining -- comparing
TEAM_ID between consecutive seasons already reflects the team a player ended
up on.
"""

from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptstats

OUTPUT_CSV = "player_history_stats.csv"
START_SEASON = "2013-14"  # first season with real (non-zero) tracking data
REQUEST_TIMEOUT = 60  # seconds
SLEEP_BETWEEN_CALLS = 1.0  # be polite to stats.nba.com

KEYS = ["PLAYER_ID", "SEASON"]

# Columns kept from each pull (plus KEYS, added automatically below).
ADV_COLS = ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "GP", "MIN", "USG_PCT", "TS_PCT"]
CATCH_SHOOT_COLS = ["CATCH_SHOOT_FGA", "CATCH_SHOOT_FG_PCT"]
PULL_UP_COLS = ["PULL_UP_FGA", "PULL_UP_FG_PCT"]
POSSESSIONS_COLS = ["TOUCHES", "TIME_OF_POSS", "AVG_SEC_PER_TOUCH"]


def recent_seasons(start_season: str, latest_start_year: int) -> list[str]:
    """Return season labels like '2013-14', ..., newest last."""
    start_year = int(start_season.split("-")[0])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, latest_start_year + 1)]


def latest_completed_season_start_year(today: pd.Timestamp) -> int:
    """Start year of the most recently completed regular season.

    NBA seasons run ~Oct->Apr. By July of a given year, the season that began
    the previous October has finished, so its start year is `year - 1`.
    From October onward, the season starting that year is underway.
    """
    return today.year if today.month >= 10 else today.year - 1


def fetch_advanced(season: str) -> pd.DataFrame:
    """Fetch one season of per-player Advanced stats (USG_PCT, TS_PCT, AGE, ...)."""
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        timeout=REQUEST_TIMEOUT,
    )
    df = resp.get_data_frames()[0]
    df.insert(0, "SEASON", season)
    return df[KEYS + ADV_COLS]


def fetch_tracking(season: str, pt_measure_type: str, keep_cols: list[str]) -> pd.DataFrame:
    """Fetch one season of per-player tracking stats for a given PtMeasureType."""
    resp = leaguedashptstats.LeagueDashPtStats(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team="Player",
        per_mode_simple="PerGame",
        pt_measure_type=pt_measure_type,
        timeout=REQUEST_TIMEOUT,
    )
    df = resp.get_data_frames()[0]
    df.insert(0, "SEASON", season)
    return df[KEYS + keep_cols]


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch + merge Advanced and all three tracking pulls for one season."""
    adv = fetch_advanced(season)
    time.sleep(SLEEP_BETWEEN_CALLS)
    catch_shoot = fetch_tracking(season, "CatchShoot", CATCH_SHOOT_COLS)
    time.sleep(SLEEP_BETWEEN_CALLS)
    pull_up = fetch_tracking(season, "PullUpShot", PULL_UP_COLS)
    time.sleep(SLEEP_BETWEEN_CALLS)
    possessions = fetch_tracking(season, "Possessions", POSSESSIONS_COLS)
    time.sleep(SLEEP_BETWEEN_CALLS)

    data = adv
    for df in (catch_shoot, pull_up, possessions):
        data = data.merge(df, on=KEYS, how="inner", validate="one_to_one")
    return data


def main() -> None:
    today = pd.Timestamp.today()
    latest = latest_completed_season_start_year(today)
    seasons = recent_seasons(START_SEASON, latest)

    print(f"Pulling {len(seasons)} seasons: {seasons[0]} ... {seasons[-1]}")

    frames: list[pd.DataFrame] = []
    for season in seasons:
        print(f"  fetching {season} (advanced + tracking) ...", flush=True)
        frames.append(fetch_season(season))

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(data)} rows x {data.shape[1]} cols to {OUTPUT_CSV}")
    print(f"Players per season: {data.groupby('SEASON')['PLAYER_ID'].nunique().to_dict()}")

    print("\n=== Column names ===")
    print(list(data.columns))

    print("\n=== First 10 rows ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(data.head(10))


if __name__ == "__main__":
    main()
