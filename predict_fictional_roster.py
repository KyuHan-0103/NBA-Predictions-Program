"""Interactively build a fictional 5-15 player roster and predict its record.

Input method
------------
Players are looked up by name against every player-season pulled by
pull_player_seasons_all.py (1996-97 onward -- see that script for why that's
the floor). Identical players from different seasons are different roster
entries (2019-20 Stephen Curry != 2022-23 Stephen Curry) because they're
looked up and validated as distinct rows keyed on (PLAYER_ID, SEASON).

For each player the user picks either:
  * "Prime" -- the statistically best season of that player's career, defined
    as the eligible season (see below) with the highest PIE (Player Impact
    Estimate). PIE is stats.nba.com's own single-number, box-score-derived
    summary of a player's per-game impact, already pulled alongside the other
    Advanced columns -- a reasonable, no-extra-computation stand-in for "best
    season" that needs no external metric. It is NOT era- or pace-adjusted,
    so it's a heuristic, not a ground truth (see README Limitations).
  * A specific season (e.g. "2019-20").

Eligibility ("Prime" candidates, and any specific season picked) reuses
aggregate_team.py's own filter (GP >= MIN_GP, MIN >= MIN_MPG) -- the
aggregation contract in CLAUDE.md requires the eligibility filter already be
applied before a roster reaches aggregate_team(), so it's applied here, at
input time, not left for aggregate_team to assume. A season is also rejected
if its Advanced columns aren't finite or are all zero (see _is_usable) --
defensive, since pull_player_seasons_all.py hasn't found this to actually
happen at the 1996-97 floor, but the aggregation contract forbids NaN/inf and
this project has already hit one silent-zero trap (pull_player_history.py's
pre-2013-14 tracking columns), so a cross-check costs nothing.

If a player isn't found, didn't play in the given season, or the season isn't
one this project has data for, the tool reports why and asks for another
player -- it never silently drops or substitutes.

This script only builds a roster and scores it -- it does not modify
aggregate_team.py's aggregation logic.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from aggregate_team import (
    DEF_TARGET,
    GAMES_PER_SEASON,
    MIN_GP,
    MIN_MPG,
    TEAM_CSV,
    TEAM_DROP_COLS,
    TEAM_TARGET,
    aggregate_team,
    record,
)
from train_model import build_model

PLAYER_CSV = "player_all_seasons.csv"
MIN_ROSTER = 5
MAX_ROSTER = 15
# Advanced columns checked for the NaN/inf/all-zero aggregation-contract guard.
_ADV_CHECK_COLS = ["MIN", "POSS", "PACE", "OREB_PCT", "DREB_PCT", "PIE"]
_SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")


def load_player_pool(csv_path: str = PLAYER_CSV) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, dtype={"SEASON": str})
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{csv_path} not found -- run `python pull_player_seasons_all.py` first."
        ) from exc


def eligible_mask(pool: pd.DataFrame) -> pd.Series:
    """GP/MIN thresholds (aggregate_team.filter_players) + the NaN/inf/all-zero
    guard the aggregation contract requires."""
    vals = pool[_ADV_CHECK_COLS].to_numpy(dtype=float)
    finite = np.isfinite(vals).all(axis=1)
    not_all_zero = (vals != 0).any(axis=1)
    thresholds = (pool["GP"] >= MIN_GP) & (pool["MIN"] >= MIN_MPG)
    return thresholds & finite & not_all_zero


def normalize_season(season: str) -> str | None:
    """'2019-20' -> '2019-20'; rejects malformed or non-consecutive labels."""
    m = _SEASON_RE.match(season.strip())
    if not m:
        return None
    start = int(m.group(1))
    if str(start + 1)[-2:] != m.group(2):
        return None
    return f"{m.group(1)}-{m.group(2)}"


def _fold(name: str) -> str:
    """Case/diacritic/punctuation-insensitive form for matching: 'Dončić' ->
    'doncic', "Shaquille O'Neal" -> 'shaquille oneal' -- so a plain-ASCII,
    no-punctuation query still finds accented or apostrophe'd NBA API names."""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    no_punct = re.sub(r"[^a-z0-9 ]", "", stripped.lower())
    return re.sub(r"\s+", " ", no_punct).strip()


def find_player_candidates(pool: pd.DataFrame, query: str) -> pd.DataFrame:
    """Exact (case/diacritic-insensitive) match if unique, else substring matches."""
    names = pool[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()
    folded = names["PLAYER_NAME"].map(_fold)
    q = _fold(query)
    if not q:
        return names.iloc[0:0]

    exact = names[folded == q]
    if len(exact) >= 1:
        return exact
    return names[folded.str.contains(q, regex=False)]


def eligible_seasons_for_player(pool_elig: pd.DataFrame, player_id) -> pd.DataFrame:
    return pool_elig[pool_elig["PLAYER_ID"] == player_id].sort_values("SEASON")


def prime_season_row(pool_elig: pd.DataFrame, player_id) -> pd.Series | None:
    """The eligible season with the highest PIE -- see module docstring."""
    rows = eligible_seasons_for_player(pool_elig, player_id)
    if rows.empty:
        return None
    return rows.loc[rows["PIE"].idxmax()]


def season_row(
    pool: pd.DataFrame, pool_elig: pd.DataFrame, player_id, season: str
) -> tuple[pd.Series | None, str]:
    """Look up one (player_id, season) row, or a specific reason it's unusable."""
    normalized = normalize_season(season)
    if normalized is None:
        return None, f"'{season}' isn't a valid season label (expected e.g. '2019-20')."
    season = normalized

    if season not in set(pool["SEASON"]):
        return None, (
            f"No data for the {season} season (this project's data covers "
            f"{pool['SEASON'].min()} through {pool['SEASON'].max()})."
        )

    player_rows = pool[pool["PLAYER_ID"] == player_id]
    if season not in set(player_rows["SEASON"]):
        return None, f"This player did not play in the {season} season."

    elig_rows = pool_elig[(pool_elig["PLAYER_ID"] == player_id) & (pool_elig["SEASON"] == season)]
    if elig_rows.empty:
        return None, (
            f"This player's {season} stat line doesn't meet the minimum "
            f"eligibility threshold (GP >= {MIN_GP}, MIN >= {MIN_MPG} mpg) or "
            f"contains unusable (NaN/inf/all-zero) data."
        )
    return elig_rows.iloc[0], ""


def interactive_build_roster(pool: pd.DataFrame) -> pd.DataFrame:
    """Prompt on stdin for 5-15 (player, season) picks; return the roster rows."""
    pool_elig = pool[eligible_mask(pool)]
    roster_rows: list[pd.Series] = []
    chosen: set[tuple] = set()

    print(f"Build a fictional roster ({MIN_ROSTER}-{MAX_ROSTER} players).")
    print("Enter a player name, or 'done' once you have at least 5.\n")

    while len(roster_rows) < MAX_ROSTER:
        prompt = f"[{len(roster_rows)}/{MAX_ROSTER}] Player name"
        if len(roster_rows) >= MIN_ROSTER:
            prompt += " (or 'done' to finish)"
        name = input(prompt + ": ").strip()

        if name.lower() in ("done", "finish"):
            if len(roster_rows) < MIN_ROSTER:
                print(f"  [error] Need at least {MIN_ROSTER} players (have {len(roster_rows)}).\n")
                continue
            break
        if not name:
            continue

        candidates = find_player_candidates(pool, name)
        if candidates.empty:
            print(f"  [error] Player '{name}' not found. Try another player.\n")
            continue

        if len(candidates) > 1:
            print("  Multiple matches:")
            for i, (_, r) in enumerate(candidates.iterrows(), 1):
                print(f"    {i}. {r['PLAYER_NAME']}")
            pick = input("  Enter a number (blank to cancel): ").strip()
            if not pick.isdigit() or not (1 <= int(pick) <= len(candidates)):
                print("  [error] No player selected. Try another player.\n")
                continue
            chosen_row = candidates.iloc[int(pick) - 1]
        else:
            chosen_row = candidates.iloc[0]
        player_id, player_name = chosen_row["PLAYER_ID"], chosen_row["PLAYER_NAME"]

        mode = input(f"  '{player_name}' -- (P)rime season or specific (S)eason? [P/s]: ").strip().lower()
        if mode in ("", "p", "prime"):
            row = prime_season_row(pool_elig, player_id)
            if row is None:
                print(
                    f"  [error] No qualifying season for {player_name} "
                    f"(GP >= {MIN_GP}, MIN >= {MIN_MPG} mpg). Try another player.\n"
                )
                continue
        else:
            season = input("  Season (e.g. 2019-20): ").strip()
            row, err = season_row(pool, pool_elig, player_id, season)
            if row is None:
                print(f"  [error] {err} Try another player.\n")
                continue

        key = (player_id, row["SEASON"])
        if key in chosen:
            print(f"  [error] {player_name} ({row['SEASON']}) is already on the roster.\n")
            continue

        chosen.add(key)
        roster_rows.append(row)
        print(f"  Added {player_name} ({row['SEASON']}). Roster: {len(roster_rows)}/{MAX_ROSTER}\n")

    return pd.DataFrame(roster_rows).reset_index(drop=True)


def feature_columns(teams_all: pd.DataFrame) -> list[str]:
    """Same feature set aggregate_team.py's own validation uses: team_season_stats.csv
    columns minus identifiers/GP/target, minus DEF_RATING (see aggregate_team
    module docstring -- no honest analogue for a fictional roster)."""
    return [c for c in teams_all.columns if c not in TEAM_DROP_COLS + [TEAM_TARGET, DEF_TARGET]]


def predict_roster(roster: pd.DataFrame) -> float:
    """Aggregate the roster, fit ridge on all available team-seasons (no
    holdout -- there's no real season to validate a fictional team against,
    so the final model uses every row for the best fit), and predict W_PCT."""
    teams_all = pd.read_csv(TEAM_CSV)
    feature_cols = feature_columns(teams_all)

    agg = aggregate_team(roster, feature_cols)

    model = build_model()
    model.fit(teams_all[feature_cols], teams_all[TEAM_TARGET])
    pred = float(model.predict(agg.to_frame().T)[0])

    print("\n=== Fictional roster ===")
    print(roster[["PLAYER_NAME", "SEASON"]].to_string(index=False))

    print(f"\nPredicted record over {GAMES_PER_SEASON} games: {record(pred)}  (W_PCT {pred:.3f})")
    low_conf = agg.attrs.get("low_confidence_cols", ())
    if low_conf:
        print(f"Low-confidence (approximated, not box-score-derived): {', '.join(low_conf)}")
    print("Note: DEF_RATING is dropped from this prediction -- see README Limitations.")
    return pred


def main() -> None:
    pool = load_player_pool()
    roster = interactive_build_roster(pool)
    predict_roster(roster)


if __name__ == "__main__":
    main()
