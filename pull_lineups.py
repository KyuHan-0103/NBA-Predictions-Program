"""Pull 5-man lineup stats (Base + Advanced) for the same seasons as
team_season_stats.csv, for use as ground truth in aggregate_team.py's
5-man-unit validation (see CLAUDE.md's aggregation contract).

Season floor (confirmed empirically, not assumed)
---------------------------------------------------
Querying LeagueDashLineups (GroupQuantity=5, Advanced measure type) returns
an EMPTY frame -- zero rows, not zeroed/garbage columns -- for every season
checked through 2006-07, and real, non-zero data from 2007-08 onward. That's
a later floor than pull_player_seasons_all.py's 1996-97 floor for individual
player Advanced stats, but a clean empty-frame failure the same way: no
silent-zero trap found. This script only pulls 2014-15 onward anyway (to
match team_season_stats.csv's range), so the 2007-08 floor is documented
here for completeness rather than exercised by START_SEASON below.

Row-count cap (verified, not assumed)
----------------------------------------
LeagueDashLineups caps each (season, measure_type) response at 2000 rows.
Confirmed two ways: the smallest-sample rows in a full response cluster at
an identical, suspiciously round MIN (~12.0) rather than tailing off
smoothly, and independently pulling Base vs. Advanced for the same season
returns overlapping-but-not-identical 2000-row sets (55 of 2000 lineups
differed in one spot-checked season, 2023-24) -- both point to a hard
server-side truncation in the low-minutes tail, not a real gap in the
underlying data. The inner merge below therefore silently drops the ~2-3%
of lineups that land on opposite sides of that cut between the two calls.
This only touches the extreme low-minutes tail (garbage-time-only
combinations); the printed POSS distribution in main() exists precisely so
a validation possession floor can be chosen well clear of it, rather than
guessed.

Units (confirmed against team_season_stats.csv, not assumed)
----------------------------------------------------------------
Lineup OFF_RATING/DEF_RATING come back on the same per-100-possessions scale
as team_season_stats.csv's columns of the same name (both ~100-130) -- no
unit conversion applied here. Per CLAUDE.md's aggregation contract, units
must agree with the team CSV before a column is trusted downstream; this
was checked, not assumed. Lineup POSS, like every other counting stat here,
arrives as a season total (not per-game, not per-100) and is divided by GP
below along with the Base box-score columns, exactly like
pull_player_seasons_all.py.

Lineup identity
------------------
The endpoint does not return five separate player-ID columns. GROUP_ID comes
back as a single string of dash-joined player IDs with a leading and
trailing dash, e.g. "-203484-203932-203999-1627750-1629008-" -- confirmed by
inspecting a live response, not assumed from docs. It's parsed below into
five integer columns (PLAYER_ID_1..PLAYER_ID_5, sorted ascending for a
canonical, order-independent lineup identity).

This script only pulls and stores data -- no modeling.
"""

from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashlineups

from aggregate_team import ADDITIVE_COLS

OUTPUT_CSV = "lineup_season_stats.csv"
START_SEASON = "2014-15"  # match team_season_stats.csv's range
GROUP_QUANTITY = 5  # 5-man lineups
N_PLAYERS = 5
REQUEST_TIMEOUT = 60  # seconds
SLEEP_BETWEEN_CALLS = 1.0  # be polite to stats.nba.com

KEYS = ["SEASON", "GROUP_ID", "TEAM_ID"]
# Explicit allow-lists (not "keep all, then drop") -- leakage columns the
# endpoint also returns (W, L, W_PCT, PLUS_MINUS, NET_RATING, and the
# E_-prefixed estimated on/off ratings) are simply never selected, matching
# the exclusions CLAUDE.md requires project-wide.
BASE_EXTRA = ["GROUP_NAME", "TEAM_ABBREVIATION", "GP", "MIN"] + ADDITIVE_COLS
# Advanced columns needed both for the required minimum (POSS, OFF_RATING,
# DEF_RATING) and for aggregate_team.py's 5-man validation (Task 1C), which
# needs every rate aggregate_team derives available as ground truth too.
ADV_EXTRA = [
    "POSS", "OFF_RATING", "DEF_RATING",
    "AST_TO", "AST_RATIO", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT", "EFG_PCT", "PACE",
]
# Season-total counting stats that must be divided by GP to become per-game.
_PER_GAME = ADDITIVE_COLS + ["MIN", "POSS"]
# Final column order: identity first, then the required minimum, then the rest.
_ID_COLS = [f"PLAYER_ID_{i + 1}" for i in range(N_PLAYERS)]
FINAL_COLS = (
    ["SEASON"] + _ID_COLS
    + ["TEAM_ID", "TEAM_ABBREVIATION", "GROUP_NAME", "GP", "MIN", "POSS", "OFF_RATING", "DEF_RATING"]
    + [c for c in ADDITIVE_COLS if c not in ("MIN", "POSS")]
    + [c for c in ADV_EXTRA if c not in ("POSS", "OFF_RATING", "DEF_RATING")]
)


def recent_seasons(start_season: str, latest_start_year: int) -> list[str]:
    """Return season labels like '2014-15', ..., newest last."""
    start_year = int(start_season.split("-")[0])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, latest_start_year + 1)]


def latest_completed_season_start_year(today: pd.Timestamp) -> int:
    """Start year of the most recently completed regular season.

    NBA seasons run ~Oct->Apr. By July of a given year, the season that began
    the previous October has finished, so its start year is `year - 1`.
    From October onward, the season starting that year is underway.
    """
    return today.year if today.month >= 10 else today.year - 1


def parse_group_id(group_id: str) -> list[int]:
    """"-203484-203932-203999-1627750-1629008-" -> 5 sorted player IDs."""
    ids = sorted(int(p) for p in group_id.split("-") if p)
    if len(ids) != N_PLAYERS:
        raise ValueError(f"expected {N_PLAYERS} player IDs in GROUP_ID {group_id!r}, got {len(ids)}")
    return ids


def fetch_measure(season: str, measure_type: str) -> pd.DataFrame:
    """Fetch one season of 5-man lineup stats for a measure type."""
    resp = leaguedashlineups.LeagueDashLineups(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        measure_type_detailed_defense=measure_type,
        group_quantity=GROUP_QUANTITY,
        timeout=REQUEST_TIMEOUT,
    )
    df = resp.get_data_frames()[0]
    df.insert(0, "SEASON", season)
    return df


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch + merge Base and Advanced for one season's 5-man lineups,
    parse lineup identity, and convert totals to per-game."""
    base = fetch_measure(season, "Base")[KEYS + BASE_EXTRA]
    time.sleep(SLEEP_BETWEEN_CALLS)
    adv = fetch_measure(season, "Advanced")[KEYS + ADV_EXTRA]
    time.sleep(SLEEP_BETWEEN_CALLS)

    df = base.merge(adv, on=KEYS, how="inner", validate="one_to_one")
    df = df[df["GP"] > 0].copy()  # guard the per-game division below

    ids_df = pd.DataFrame(
        df["GROUP_ID"].apply(parse_group_id).tolist(), columns=_ID_COLS, index=df.index,
    )
    df = pd.concat([df.drop(columns=["GROUP_ID"]), ids_df], axis=1)

    df[_PER_GAME] = df[_PER_GAME].div(df["GP"], axis=0)
    return df[FINAL_COLS]


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

    print(f"\nSaved {len(data)} lineup-seasons x {data.shape[1]} cols to {OUTPUT_CSV}")

    print("\n=== Column names ===")
    print(list(data.columns))

    print("\n=== First 10 rows ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(data.head(10))

    print("\n=== Row counts per season (2000-row cap per measure-type call -- see module docstring) ===")
    print(data.groupby("SEASON").size().to_string())

    # Season-total POSS (not the per-game column stored above) is the right
    # axis for choosing a validation possession floor -- it reflects how much
    # game action actually backs a lineup's stat line, which per-game POSS
    # does not (a 1-game cameo and a full-season starter can share a per-game
    # rate while differing enormously in reliability).
    total_poss = data["POSS"] * data["GP"]
    print("\n=== POSS-per-lineup distribution (season total, for choosing a possession floor) ===")
    print(total_poss.describe().to_string())
    print("\nPercentiles:")
    print(total_poss.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())


if __name__ == "__main__":
    main()
