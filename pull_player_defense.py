"""Pull per-player *individual* defensive descriptors (bio, closest-defender
tracking, hustle) for lineup_defense_model.py -> player_defense_stats.csv.

Why a separate pull
-------------------
player_all_seasons.csv already carries every player's own box score (BLK, STL,
DREB, PF, BLKA, PFD) plus OREB_PCT/DREB_PCT. What it does not carry is the
non-box-score side of individual defense: how big the player is, who he was
asked to guard, and how hard he worked off the ball. Those live on three other
endpoints, pulled here.

Every column below describes the PLAYER, never the team-while-he-was-on-court.
On/off-style columns the same endpoints happily return (NET_RATING on the bio
endpoint, for instance) are deliberately never selected -- see
lineup_defense_model.py's forbidden-feature rule, which exists to keep the
circularity documented in README.md's DEF_RATING limitation out of the model.

Season floors (confirmed empirically, not assumed)
-------------------------------------------------
* Bio (LeagueDashPlayerBioStats): available for every season checked, incl.
  1996-97. Height/weight/age are roster facts, not tracked events.
* Closest-defender tracking (LeagueDashPtDefend): real data from 2013-14 on;
  2012-13 and earlier return an EMPTY frame (0 rows, not zeroed columns), the
  same clean failure mode pull_player_seasons_all.py documents for Advanced.
  This is the SportVU tracking floor, matching pull_player_history.py.
* Hustle (LeagueHustleStatsPlayer): 2015-16 is a TRAP -- it returns a
  non-empty frame (147 players) whose median G is 1, because the NBA only
  tracked hustle stats for a handful of games at the end of that season. Taken
  at face value it looks like real data; used as a season line it is garbage.
  The first fully-tracked season is 2016-17 (485 players, median G = 61), so
  HUSTLE_START_SEASON is 2016-17 and hustle columns are left as NaN (never
  0.0) for earlier seasons. Downstream code must treat them as missing, not
  as "this player recorded no deflections".

Units
-----
Counting stats arrive as season totals and are divided by games played to
per-game, matching player_all_seasons.csv / lineup_season_stats.csv. Rates
(D_FG_PCT, NORMAL_FG_PCT, PCT_PLUSMINUS) are already rates and are left as-is.
FREQ is not kept: for defense_category="Overall" it is identically 1.0 for
every player (verified), so it carries no information.

One-row-per-player-season is not free (found, not assumed)
----------------------------------------------------------
LeagueDashPtDefend occasionally returns TWO rows for one player-season -- e.g.
2013-14 Jordan Hamilton appears as "F" (37 GP) and "F-G" (20 GP), 481 rows for
480 unique players -- because the endpoint splits a player whose listed
position changed mid-season. Rows like that are combined below by summing the
season totals and recomputing the rates from the combined totals (keeping the
position label from the longer stint), so the output really is one row per
player-season. Side effect worth knowing: for those few players the defensive
totals here cover the FULL season, while their player_all_seasons.csv row
covers only their final team's stint (the traded-player caveat documented
there).

This script only pulls and stores data -- no modeling.
"""

from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerbiostats,
    leaguedashptdefend,
    leaguehustlestatsplayer,
)

OUTPUT_CSV = "player_defense_stats.csv"
# 2013-14: the closest-defender tracking floor (see module docstring). Earlier
# seasons would carry bio columns only, which lineup_defense_model.py can't use
# on its own, so the pull starts here rather than at the bio floor.
START_SEASON = "2013-14"
HUSTLE_START_SEASON = "2016-17"  # first fully-tracked hustle season
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_CALLS = 1.0  # be polite to stats.nba.com

KEYS = ["PLAYER_ID", "SEASON"]

# Roster facts (bio). NET_RATING / TS_PCT / USG_PCT are also returned by this
# endpoint and are deliberately not selected (on-court team context / offense).
BIO_KEEP = ["PLAYER_NAME", "AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT"]
# Closest-defender tracking: what happened to shots this player defended.
DEFEND_KEEP = [
    "PLAYER_POSITION", "D_FGM", "D_FGA", "D_FG_PCT", "NORMAL_FG_PCT", "PCT_PLUSMINUS",
]
DEFEND_PER_GAME = ["D_FGM", "D_FGA"]  # season totals -> per game
# Hustle: off-box-score defensive effort. Offense-side hustle columns
# (SCREEN_ASSISTS, OFF_BOXOUTS, OFF_LOOSE_BALLS_RECOVERED) are not selected.
HUSTLE_KEEP = [
    "CONTESTED_SHOTS", "CONTESTED_SHOTS_2PT", "CONTESTED_SHOTS_3PT",
    "DEFLECTIONS", "CHARGES_DRAWN", "DEF_BOXOUTS", "DEF_LOOSE_BALLS_RECOVERED",
]
HUSTLE_PER_GAME = HUSTLE_KEEP  # all counting stats -> per game


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


def fetch_bio(season: str) -> pd.DataFrame:
    """Height / weight / age, one row per player-season."""
    df = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_simple="Totals",
        timeout=REQUEST_TIMEOUT,
    ).get_data_frames()[0]
    df.insert(0, "SEASON", season)
    out = df[KEYS + BIO_KEEP].copy()
    # PLAYER_WEIGHT comes back as a string ("195"), and is blank for a few
    # historic players -> coerce rather than carry object dtype downstream.
    out["PLAYER_WEIGHT"] = pd.to_numeric(out["PLAYER_WEIGHT"], errors="coerce")
    # Height/weight/age don't change between stints, so a duplicated player
    # (same reason as the tracking split above) is safe to collapse.
    return out.drop_duplicates(subset=KEYS, keep="first")


def _combine_split_defend_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a player-season split across two position labels into one row.

    Totals are summed and the rates recomputed from the combined totals (a
    D_FGA-weighted average for NORMAL_FG_PCT, the league baseline for the shots
    he defended); the position label comes from the longer stint. A no-op for
    the ~99.8% of player-seasons that arrive as a single row.
    """
    if not df.duplicated(subset=KEYS).any():
        return df

    def collapse(g: pd.DataFrame) -> pd.Series:
        totals = g[["GP", "D_FGM", "D_FGA"]].sum()
        d_fg_pct = totals["D_FGM"] / totals["D_FGA"] if totals["D_FGA"] else 0.0
        normal = (
            float((g["NORMAL_FG_PCT"] * g["D_FGA"]).sum() / g["D_FGA"].sum())
            if g["D_FGA"].sum() else 0.0
        )
        return pd.Series({
            "PLAYER_POSITION": g.loc[g["GP"].idxmax(), "PLAYER_POSITION"],
            "GP": totals["GP"],
            "D_FGM": totals["D_FGM"],
            "D_FGA": totals["D_FGA"],
            "D_FG_PCT": d_fg_pct,
            "NORMAL_FG_PCT": normal,
            "PCT_PLUSMINUS": d_fg_pct - normal,
        })

    return df.groupby(KEYS, as_index=False).apply(collapse, include_groups=False)


def fetch_defend(season: str) -> pd.DataFrame:
    """Closest-defender tracking: opponent FG% on shots this player defended."""
    df = leaguedashptdefend.LeagueDashPtDefend(
        season=season,
        season_type_all_star="Regular Season",
        defense_category="Overall",
        per_mode_simple="Totals",
        timeout=REQUEST_TIMEOUT,
    ).get_data_frames()[0]
    if df.empty:  # pre-2013-14: empty frame, not an error (see docstring)
        raise ValueError(f"no closest-defender tracking data for {season}")
    df.insert(0, "SEASON", season)
    df = df.rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"})
    df = df[df["GP"] > 0].copy()  # guard the per-game division below
    df = _combine_split_defend_rows(df[KEYS + DEFEND_KEEP + ["GP"]])
    df[DEFEND_PER_GAME] = df[DEFEND_PER_GAME].div(df["GP"], axis=0)
    return df[KEYS + DEFEND_KEEP]


def fetch_hustle(season: str) -> pd.DataFrame:
    """Hustle stats (2016-17 onward -- see HUSTLE_START_SEASON)."""
    df = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_time="Totals",
        timeout=REQUEST_TIMEOUT,
    ).get_data_frames()[0]
    if df.empty:
        raise ValueError(f"no hustle data for {season}")
    df.insert(0, "SEASON", season)
    df = df[df["G"] > 0].copy()  # guard the per-game division below
    # Same one-row-per-player-season guard as the tracking pull: sum any split
    # rows' totals (all hustle columns kept here are counting stats) first.
    df = df.groupby(KEYS, as_index=False)[HUSTLE_KEEP + ["G"]].sum()
    df[HUSTLE_PER_GAME] = df[HUSTLE_PER_GAME].div(df["G"], axis=0)
    return df[KEYS + HUSTLE_KEEP]


def fetch_season(season: str) -> pd.DataFrame:
    """Bio + tracking (+ hustle when the season has it), merged per player.

    Left-joined onto the tracking frame: a player with no defended-shot row
    (didn't play) has nothing to say about defense here. Hustle is left NaN --
    not 0 -- for seasons before HUSTLE_START_SEASON and for players missing
    from an otherwise-tracked season.
    """
    defend = fetch_defend(season)
    time.sleep(SLEEP_BETWEEN_CALLS)
    bio = fetch_bio(season)
    time.sleep(SLEEP_BETWEEN_CALLS)

    out = defend.merge(bio, on=KEYS, how="left", validate="one_to_one")

    if season >= HUSTLE_START_SEASON:
        hustle = fetch_hustle(season)
        time.sleep(SLEEP_BETWEEN_CALLS)
        out = out.merge(hustle, on=KEYS, how="left", validate="one_to_one")
    else:
        for c in HUSTLE_KEEP:
            out[c] = pd.NA

    return out


def main() -> None:
    today = pd.Timestamp.today()
    seasons = recent_seasons(START_SEASON, latest_completed_season_start_year(today))

    print(f"Pulling {len(seasons)} seasons: {seasons[0]} ... {seasons[-1]}")
    print(f"Hustle columns pulled from {HUSTLE_START_SEASON} onward (NaN before)\n")

    frames: list[pd.DataFrame] = []
    for season in seasons:
        print(f"  fetching {season} ...", flush=True)
        frames.append(fetch_season(season))

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(data)} player-seasons x {data.shape[1]} cols to {OUTPUT_CSV}")

    print("\n=== Column names ===")
    print(list(data.columns))

    print("\n=== Rows per season, and non-null share of the optional blocks ===")
    per_season = data.groupby("SEASON").agg(
        rows=("PLAYER_ID", "size"),
        height_notnull=("PLAYER_HEIGHT_INCHES", lambda s: s.notna().mean()),
        hustle_notnull=("DEFLECTIONS", lambda s: s.notna().mean()),
    )
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(per_season.to_string())

    print("\n=== First 5 rows ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(data.head(5))


if __name__ == "__main__":
    main()
