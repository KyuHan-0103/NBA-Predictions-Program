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
     * USAGE CONSERVATION (POSS, PACE, and the shot/turnover/assist volume
       stats): summing 5 players' own box-score production has no boundary --
       nothing stops 5 high-usage players from implying more team possessions
       than a real game contains. POSS_target (a minutes-weighted average of
       the players' own PACE, which tracks real team PACE closely -- ~1.4%
       mean abs error, corr 0.985, checked against every real team this
       season) anchors the roster to a realistic number of possessions; every
       usage-linked stat is scaled by POSS_target/POSS_raw, and POSS itself is
       then *derived* from the scaled totals (not set independently), so the
       possession denominator always matches the scaled numerators exactly --
       OFF_RATING and TM_TOV_PCT come out identical whether conservation is on
       or off (see aggregate_team's conserve_usage flag).
     * DERIVED rates (OFF_RATING, EFG_PCT, AST_TO, AST_RATIO, TM_TOV_PCT):
       recomputed from the aggregated (and usage-scaled) box-score totals with
       standard formulas.
     * OPPONENT-DEPENDENT rates (OREB_PCT, DREB_PCT): can't be rebuilt from an
       offensive box score, so they're approximated from the players' own
       rebound *shares* -- the five on-court players' individual rates sum to
       the team's, so they scale up by the ~5 players sharing the floor. This
       is the softest estimate here -- see the validation output.

DEF_RATING is dropped entirely, not approximated (see "Limitations" in
README.md). It used to be filled in as a minutes-weighted average of the
players' own real on-court DEF_RATING, and that measurably beat both
alternatives tried (a personal-box-score composite, or just dropping it) --
but only because it borrows real, current-season, opponent-dependent data
that a genuinely fictional or cross-era roster could never have. Keeping it
would make the aggregation pipeline's accuracy depend on data the project's
actual goal (fictional/historic rosters) can't supply, so it's dropped for
consistency even though that costs some measured accuracy on real rosters.

Validation
----------
For each real team this season, take its qualifying players, aggregate them, and
compare the synthetic line to the team's actual line from team_season_stats.csv.

The per-feature error-impact diagnostics (perturbation_impact() and the
deprecated MAPE x coefficient ranking) live in perturbation_tests.py. main()
below still runs and prints them exactly as before, via
run_perturbation_tests(); that file can also be run on its own.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from train_model import build_model, extrapolation_report, fit_extrapolation_guard

TEAM_CSV = "team_season_stats.csv"
TEAM_DROP_COLS = ["SEASON", "TEAM_ID", "TEAM_NAME", "GP"]
TEAM_TARGET = "W_PCT"
GAMES_PER_SEASON = 82  # for expressing W_PCT as a win-loss record

# --- 5-man lineup validation (pull_lineups.py / pull_player_seasons_all.py) ---
LINEUP_CSV = "lineup_season_stats.csv"
PLAYER_SEASONS_CSV = "player_all_seasons.csv"
LINEUP_PLAYER_ID_COLS = [f"PLAYER_ID_{i + 1}" for i in range(5)]
# Total-season possessions floor for a lineup to count as ground truth here.
# Chosen from pull_lineups.py's printed distribution (23,470 lineup-seasons,
# 2014-15..2025-26): the median lineup-season has only ~45 total possessions
# (most rosters run dozens of small-sample bench combinations), while 250
# sits around the 95th percentile -- roughly each team's handful of
# most-used units per season. Below 250, a "lineup" is mostly a few-game
# garbage-time cameo, not a real rotation unit worth validating against.
LINEUP_POSS_FLOOR = 250
# Dropped from THIS script's own feature set (see module docstring): real,
# opponent-dependent, and unavailable for a genuinely fictional/cross-era
# roster. train_model.py's own real-team model is unaffected by this.
DEF_TARGET = "DEF_RATING"

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
# Usage-conservation family: scaled together by POSS_target/POSS_raw so total
# shot/turnover volume fits inside a realistic number of team possessions
# without distorting shooting splits, assist rate, or turnover rate (see
# aggregate_team). DREB is excluded -- while it's a pace dependent stat,
# it's ultimately  a function of opponent misses, not this roster's own shot volume.
# REB is recomputed as OREB+DREB afterward to
# keep the additive identity since only OREB (not DREB) is scaled here.
USAGE_SCALE_COLS = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "TOV", "OREB", "PTS", "AST"]
# Rebound shares: the 5 on-court players' individual rates sum to the team's.
SHARE_COLS = ["OREB_PCT", "DREB_PCT"]
# Player columns needed from the Advanced measure type (per-game POSS + the rates).
# DEF_RATING is deliberately not pulled -- see DEF_TARGET / module docstring.
# PACE feeds the usage-conservation scaling in aggregate_team -- it's already a
# rate (possessions per 48) so it does NOT go in _PER_GAME below.
ADV_KEEP = ["PLAYER_ID", "SEASON", "POSS", "OREB_PCT", "DREB_PCT", "PACE"]
# Counting stats that arrive as season totals and must be divided by GP.
_PER_GAME = ADDITIVE_COLS + ["MIN", "POSS"]
# Approximated (not reconstructed from a box score) -- see aggregate_team docstring.
LOW_CONFIDENCE_COLS = SHARE_COLS


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide guarding a zero denominator so ratios never produce NaN/inf.

    A zero denominator here means a degenerate roster (e.g. 0 turnovers across
    the whole team) that real eligibility-filtered rosters won't hit; `default`
    just keeps the aggregation contract (no NaN/inf) instead of raising.
    """
    return float(numerator) / float(denominator) if denominator else default


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
    """Per-game player stats incl. rebound rates (not in the curated CSV)."""
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
    conserve_usage: bool = True,
) -> pd.Series:
    """Combine 5-15 players into one team stat-line in `feature_cols` order.

    `conserve_usage=False` disables the usage-conservation scaling below (POSS
    falls back to the raw, uncapped box-score estimate) -- used only to prove
    the possession-denominated rates are invariant to the scaling, and for the
    ON-vs-OFF diagnostics in main().
    """
    n = len(players)
    if not 5 <= n <= 15:
        print(f"  [warn] roster has {n} players (expected 5-15)")

    mpg = players["MIN"].to_numpy(dtype=float)
    total_mpg = mpg.sum()
    if total_mpg <= 0:
        raise ValueError("roster has zero total minutes; cannot aggregate")
    scale = total_minutes / total_mpg      # rescale roster minutes to 240
    weights = mpg / total_mpg              # minute share, sums to 1

    agg: dict[str, float] = {}

    # Additive box-score events (rescaled sum of per-game production).
    for c in ADDITIVE_COLS:
        agg[c] = scale * players[c].to_numpy(dtype=float).sum()

    # Team minutes are game-minutes (~48), not the 240 player-minutes.
    agg["MIN"] = total_minutes / 5.0

    # Usage conservation: the ball is zero-sum, so summing 5 players' own
    # box-score volume can imply more possessions than a real team plays in a
    # game -- nothing above enforces that boundary. POSS_target anchors the
    # roster to a realistic pace: a minutes-weighted average of the players'
    # own PACE, which tracks real team PACE closely (~1.4% mean abs error,
    # corr 0.985, checked against every real team this season). POSS_raw is
    # the same total estimated the standard box-score way, uncapped.
    poss_target = float(np.average(players["PACE"].to_numpy(dtype=float), weights=mpg))
    poss_raw = agg["FGA"] + 0.44 * agg["FTA"] + agg["TOV"] - agg["OREB"]
    usage_scale = _safe_div(poss_target, poss_raw, default=1.0) if conserve_usage else 1.0
    for c in USAGE_SCALE_COLS:
        agg[c] *= usage_scale
    agg["REB"] = agg["OREB"] + agg["DREB"]  # DREB unscaled -- restore the additive identity
    # POSS is *derived* from the (now scaled) totals rather than set to
    # poss_target independently -- that keeps the possession denominator
    # exactly in step with the scaled numerators, which is what makes
    # OFF_RATING and TM_TOV_PCT below invariant to conservation on/off.
    agg["POSS"] = agg["FGA"] + 0.44 * agg["FTA"] + agg["TOV"] - agg["OREB"]
    agg["PACE"] = agg["POSS"]

    # Rates derived from the aggregated box-score totals (each guarded against a
    # zero denominator so the output never contains NaN/inf).
    agg["EFG_PCT"] = _safe_div(agg["FGM"] + 0.5 * agg["FG3M"], agg["FGA"])
    agg["AST_TO"] = _safe_div(agg["AST"], agg["TOV"])
    plays = agg["FGA"] + 0.44 * agg["FTA"] + agg["AST"] + agg["TOV"]
    agg["AST_RATIO"] = _safe_div(100.0 * agg["AST"], plays)
    # Fraction, not per-100 -- confirmed team_season_stats.csv TM_TOV_PCT is also
    # a fraction (range ~0.11-0.18), so units agree with this computation.
    agg["TM_TOV_PCT"] = _safe_div(agg["TOV"], agg["POSS"])
    agg["OFF_RATING"] = _safe_div(100.0 * agg["PTS"], agg["POSS"])

    # Opponent-dependent rates: approximated from the players' own values.
    scaled_min = weights * total_minutes  # per-player minutes, sums to 240
    team_minutes = total_minutes / 5.0    # 48 game-minutes
    # Rebound shares: the 5 on-court players' individual rates sum to the team's.
    for c in SHARE_COLS:
        agg[c] = float((players[c].to_numpy(dtype=float) * scaled_min).sum()) / team_minutes

    missing = [c for c in feature_cols if c not in agg]
    if missing:
        raise ValueError(f"aggregation did not produce required features: {missing}")

    result = pd.Series(agg, dtype=float)[feature_cols]
    if not np.isfinite(result.to_numpy()).all():
        bad = result[~np.isfinite(result.to_numpy())]
        raise ValueError(f"aggregation produced non-finite values: {bad.to_dict()}")
    # Not a data column (would violate "nothing extra" in the output contract) --
    # attrs is pandas' side-channel for exactly this kind of provenance metadata.
    result.attrs["low_confidence_cols"] = tuple(c for c in LOW_CONFIDENCE_COLS if c in feature_cols)
    result.attrs["usage_scale"] = usage_scale
    return result


# Five real, high-usage, ball-dominant guards/wings (top-10 league FGA this
# season, each a primary shot-creator) stacked on one fictional roster -- a
# stress test for usage conservation: no real team fields 5 players who all
# need this much of the shot/turnover volume at once.
STAR_STACK_NAMES = [
    "Luka Dončić",
    "Shai Gilgeous-Alexander",
    "Anthony Edwards",
    "Jalen Brunson",
    "Devin Booker",
]


def build_star_stack(players: pd.DataFrame, names: list[str] = STAR_STACK_NAMES) -> pd.DataFrame:
    """Pull a fictional high-usage roster by player name out of the season pull."""
    roster = players[players["PLAYER_NAME"].isin(names)]
    missing = set(names) - set(roster["PLAYER_NAME"])
    if missing:
        raise ValueError(f"star-stack players not found in pull: {missing}")
    return roster


def demo_star_stack(
    players: pd.DataFrame, feature_cols: list[str], model, guard: dict | None = None
) -> None:
    """Run the star-stack roster through aggregation with conservation on and
    off, and report whether it pulls an unrealistic combined shot/turnover
    volume down to something a real team could actually play.

    This roster is also the model's worst case, and the reason train_model.py
    fits log-odds: on the untransformed target the conservation-OFF line
    predicted a win rate above 1.0 (a "200-win season"). The logit makes that
    impossible, so if `guard` is supplied the extrapolation report is printed
    with the record -- a saturated 82-0 needs the flag next to it to be read
    correctly, where an impossible number spoke for itself.
    """
    roster = build_star_stack(players)
    on = aggregate_team(roster, feature_cols)
    off = aggregate_team(roster, feature_cols, conserve_usage=False)

    print(f"\n=== Star-stack stress test: {', '.join(roster['PLAYER_NAME'])} ===")
    print(f"POSS_raw (uncapped box-score estimate, conservation OFF): {off['POSS']:.1f}")
    print(f"POSS_target (pace-anchored, conservation ON):             {on['POSS']:.1f}")
    print(f"usage_scale: {on.attrs['usage_scale']:.4f}  (expect clearly < 1.0)")
    print("\nVolume pulled down by conservation (OFF = uncapped sum, ON = scaled):")
    for c in ["FGA", "FTA", "TOV", "OREB", "PTS", "AST"]:
        print(f"  {c:5s}  OFF={off[c]:8.1f}   ON={on[c]:8.1f}   ratio={on[c] / off[c]:.3f}")

    pred_on = float(model.predict(on.to_frame().T)[0])
    pred_off = float(model.predict(off.to_frame().T)[0])
    print(f"\nPredicted win record  ->  conservation ON: {record(pred_on)}   "
          f"conservation OFF: {record(pred_off)}")
    print(f"  (raw W_PCT  ON: {pred_on:.3f}   OFF: {pred_off:.3f}  -- both inside (0, 1) "
          f"by construction now; see train_model.py)")

    if guard is not None:
        print("\n" + extrapolation_report(guard, on, pred_on))


def record(wpct: float, games: int = GAMES_PER_SEASON) -> str:
    """Express a win percentage as a W-L record over `games`.

    The [0, 1] clamp is dead weight for anything the model produces now that it
    fits logit(W_PCT) -- inv_logit can't leave the interval. It stays because
    this is also called on hand-supplied and CSV-read win rates, where nothing
    guarantees that.
    """
    wins = int(round(min(max(wpct, 0.0), 1.0) * games))
    return f"{wins}-{games - wins}"


def compare(agg: pd.Series, real: pd.Series) -> pd.DataFrame:
    """Side-by-side aggregated vs. real with signed and percent error."""
    out = pd.DataFrame({"aggregated": agg, "real": real})
    out["diff"] = out["aggregated"] - out["real"]
    out["pct_err"] = 100.0 * out["diff"] / out["real"].replace(0, np.nan)
    return out


def load_lineups(poss_floor: float = LINEUP_POSS_FLOOR) -> pd.DataFrame:
    """Real 5-man lineups (from pull_lineups.py) clearing `poss_floor`
    season-total possessions -- see LINEUP_POSS_FLOOR for why."""
    lineups = pd.read_csv(LINEUP_CSV)
    total_poss = lineups["POSS"] * lineups["GP"]
    return lineups.loc[total_poss >= poss_floor].copy()


def validate_against_lineups(feature_cols: list[str]) -> tuple[pd.Series, dict]:
    """5-man ground-truth validation, as a counterpart to main()'s 15-man
    team-level validation.

    For each real 5-man lineup clearing LINEUP_POSS_FLOOR, look up its five
    players' own per-game season stats (player_all_seasons.csv), run them
    through aggregate_team() exactly as predict_fictional_roster.py would,
    and compare the synthetic line to the lineup's own real, observed line.
    Every qualifying lineup actually played those possessions together, so
    its OFF_RATING/DEF_RATING/box score are ground truth in a way the 15-man
    validation isn't: a real team's top-15 by minutes is a rotation shape,
    not the 5-player roster shape this program is actually asked to score.

    Same-minutes-basis rescale: aggregate_team()'s output represents a full,
    uninterrupted 48-minute team game (its default TOTAL_TEAM_MINUTES). A real
    lineup's own recorded line covers only the partial minutes/game it
    actually shared the floor (row["MIN"], e.g. ~15-20) before a substitution
    -- comparing the two raw would mostly measure that basis mismatch, not
    aggregation fidelity. So every count that scales with floor time (MIN,
    the box-score totals in ADDITIVE_COLS, and POSS) is scaled up by
    48/row["MIN"] before comparing -- extrapolating "what this lineup's own
    box score would look like over a full 48 minutes at its own on-court
    rate," the same target quantity aggregate_team() is built to produce.
    Rate/percentage/rating columns (OFF_RATING, EFG_PCT, AST_TO, AST_RATIO,
    TM_TOV_PCT, OREB_PCT, DREB_PCT, PACE) are already invariant to floor time
    and are left as-is. (Passing a matching total_minutes into aggregate_team
    instead would fix the box-score totals but not POSS/PACE, which
    aggregate_team derives from PACE -- already a per-48-minutes quantity --
    independent of total_minutes; rescaling the real line fixes both in one
    step.)

    A lineup is skipped (never silently included with a bad input) when:
      * one of its 5 players has no row in player_all_seasons.csv for that
        season (shouldn't happen inside the pulled range; checked
        defensively rather than assumed);
      * a player's season-long row belongs to a different TEAM_ID than the
        lineup's -- the traded-player caveat documented in
        pull_player_seasons_all.py means that row reflects only the player's
        *other* stint, not the one this lineup was actually part of;
      * any of the 5 players fails aggregate_team's own eligibility filter
        (GP >= MIN_GP, MIN >= MIN_MPG) -- the aggregation contract's own
        input precondition;
      * the lineup's own recorded MIN is <= 0 (shouldn't happen for a lineup
        with recorded possessions; checked defensively rather than assumed,
        since it's the divisor for the 48-minute rescale above).

    Returns (weighted_mape, summary): `weighted_mape` is per-feature mean
    absolute % error weighted by each lineup's season-total possessions (a
    more heavily-sampled lineup counts more); `summary` reports how many
    lineups were considered, used, and skipped for each reason above.
    """
    lineups = load_lineups()
    players_all = pd.read_csv(PLAYER_SEASONS_CSV)
    player_idx = players_all.set_index(["PLAYER_ID", "SEASON"])

    # Columns whose value scales with how long the lineup was actually on the
    # floor -- rescaled to a 48-minute-equivalent basis below (see docstring).
    # Everything else in feature_cols is already a rate/percentage/rating.
    scale_to_48 = [c for c in feature_cols if c in ADDITIVE_COLS or c in ("MIN", "POSS")]

    errs: list[pd.Series] = []
    weights: list[float] = []
    n_skipped_missing = 0
    n_skipped_traded = 0
    n_skipped_ineligible = 0
    n_skipped_zero_minutes = 0

    for _, row in lineups.iterrows():
        season = row["SEASON"]
        team_id = row["TEAM_ID"]
        player_ids = [int(row[c]) for c in LINEUP_PLAYER_ID_COLS]

        keys = [(pid, season) for pid in player_ids]
        if not all(k in player_idx.index for k in keys):
            n_skipped_missing += 1
            continue
        roster = player_idx.loc[keys].reset_index()

        if (roster["TEAM_ID"] != team_id).any():
            n_skipped_traded += 1
            continue

        eligible = filter_players(roster)
        if len(eligible) < 5:
            n_skipped_ineligible += 1
            continue

        if row["MIN"] <= 0:
            n_skipped_zero_minutes += 1
            continue

        agg = aggregate_team(eligible, feature_cols)

        # Rescale the lineup's own (partial-floor-time) line up to a
        # 48-minute-equivalent basis so it's commensurable with `agg`.
        real = row[feature_cols].astype(float).copy()
        factor = 48.0 / float(row["MIN"])
        real[scale_to_48] = real[scale_to_48] * factor

        errs.append((100.0 * (agg - real) / real.replace(0, np.nan)).abs())
        # Sample-size weight uses the lineup's real (unscaled) season-total
        # possessions, not the 48-minute extrapolation used for comparison.
        weights.append(float(row["POSS"] * row["GP"]))

    err_df = pd.concat(errs, axis=1)
    w = np.array(weights)
    weighted_mape = err_df.apply(lambda r: np.average(r, weights=w), axis=1).sort_values()

    summary = {
        "lineups_considered": len(lineups),
        "lineups_used": len(errs),
        "skipped_missing_player_season": n_skipped_missing,
        "skipped_traded_team_mismatch": n_skipped_traded,
        "skipped_ineligible_player": n_skipped_ineligible,
        "skipped_zero_minutes": n_skipped_zero_minutes,
    }
    return weighted_mape, summary


def main() -> None:
    teams_all = pd.read_csv(TEAM_CSV)
    # DEF_RATING excluded here (not from train_model.py's own feature set) --
    # see DEF_TARGET / module docstring.
    feature_cols = [c for c in teams_all.columns if c not in TEAM_DROP_COLS + [TEAM_TARGET, DEF_TARGET]]
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
    agg_line_off = aggregate_team(roster, feature_cols, conserve_usage=False)
    real_line = teams.loc[focus_id, feature_cols].astype(float)

    print(f"=== Rate invariance check: {name} (conservation ON vs OFF) ===")
    for c in ["OFF_RATING", "TM_TOV_PCT"]:
        on_v, off_v = agg_line[c], agg_line_off[c]
        print(f"  {c:12s} ON={on_v:.6f}  OFF={off_v:.6f}  diff={on_v - off_v:.2e}")
    print(f"  usage_scale (ON): {agg_line.attrs['usage_scale']:.4f}\n")

    print(f"=== Detailed validation: {name} ({len(roster)} players) ===")
    table = compare(agg_line, real_line)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(table.to_string())
    print(f"\nMean abs % error over features: {table['pct_err'].abs().mean():.2f}%")

    # --- Aggregate validation across all teams (mean abs % error per feature) ---
    errs = []
    agg_lines: dict[int, pd.Series] = {}
    agg_lines_off: dict[int, pd.Series] = {}
    usage_scales: dict[int, float] = {}
    for tid, grp in qualified.groupby("TEAM_ID"):
        grp = grp.sort_values("MIN", ascending=False).head(15)
        if tid not in teams.index or len(grp) < 5:
            continue
        a = aggregate_team(grp, feature_cols)
        agg_lines[tid] = a
        usage_scales[tid] = a.attrs["usage_scale"]
        agg_lines_off[tid] = aggregate_team(grp, feature_cols, conserve_usage=False)
        r = teams.loc[tid, feature_cols].astype(float)
        errs.append((100.0 * (a - r) / r.replace(0, np.nan)).abs())
    mape = pd.concat(errs, axis=1).mean(axis=1).sort_values()

    print(f"\n=== Validation across {len(errs)} teams: mean abs % error per feature ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(mape.to_string())
    print(f"\nOverall mean abs % error: {mape.mean():.2f}%")
    low_conf = agg_line.attrs.get("low_confidence_cols", ())
    print(f"Low-confidence (approximated, not box-score-derived): {', '.join(low_conf)}")

    # --- Diagnostic: usage_scale per real team (expect ~1.0 -- their players' ---
    # --- summed volume should already roughly fit their own pace) ---
    scale_s = pd.Series(usage_scales, name="usage_scale").rename_axis("TEAM_ID")
    scale_tbl = pd.DataFrame({"team": teams.loc[scale_s.index, "TEAM_NAME"], "usage_scale": scale_s})
    print(f"\n=== usage_scale per team (conservation ON; expect ~1.0 for real rosters) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(scale_tbl.sort_values("usage_scale").to_string(index=False))
    print(f"\nmean={scale_s.mean():.4f}  std={scale_s.std():.4f}  "
          f"min={scale_s.min():.4f}  max={scale_s.max():.4f}")

    # --- Run both lines through the ridge model: predicted win records ---
    # Train on prior seasons only, so current-season predictions are out-of-sample.
    train_df = teams_all[teams_all["SEASON"] != season]
    model = build_model()
    model.fit(train_df[feature_cols], train_df[TEAM_TARGET])

    # ids/R (every qualifying team's real line) are needed by perturbation_impact()
    # below and reused later for the full team-by-team prediction table.
    ids = list(agg_lines.keys())
    R = teams.loc[ids, feature_cols].astype(float)

    # --- Two rankings of "real" per-feature error impact, for comparison ---
    # Both live in perturbation_tests.py now (identical tests, printed here
    # exactly as before). The import is function-local on purpose: that module
    # imports this one for load_players/aggregate_team, so a module-level import
    # here would be a cycle.
    from perturbation_tests import run_perturbation_tests

    run_perturbation_tests(model, R, mape, feature_cols)

    pred_real = float(model.predict(real_line.to_frame().T)[0])
    pred_agg = float(model.predict(agg_line.to_frame().T)[0])
    actual = float(teams.loc[focus_id, TEAM_TARGET])
    print(f"\n=== {name}: predicted win record (over {GAMES_PER_SEASON} games) ===")
    print(f"  Actual               : {record(actual):>6}  (W_PCT {actual:.3f})")
    print(f"  Model on real stats  : {record(pred_real):>6}  (W_PCT {pred_real:.3f})")
    print(f"  Model on aggregated  : {record(pred_agg):>6}  (W_PCT {pred_agg:.3f})")

    # Same comparison for every team, plus the conservation-OFF counterfactual
    # (diagnostic: how much does the conservation scaling actually change the
    # aggregated win prediction, team by team?). ids/R computed above.
    A = pd.DataFrame([agg_lines[i] for i in ids], index=ids)[feature_cols]
    A_off = pd.DataFrame([agg_lines_off[i] for i in ids], index=ids)[feature_cols]
    wins = lambda wp: (np.clip(wp, 0, 1) * GAMES_PER_SEASON).round().astype(int)
    out = pd.DataFrame({
        "team": teams.loc[ids, "TEAM_NAME"].to_numpy(),
        "actual_W": wins(teams.loc[ids, TEAM_TARGET].to_numpy()),
        "pred_real_W": wins(model.predict(R)),
        "pred_agg_W": wins(model.predict(A)),
        "pred_agg_off_W": wins(model.predict(A_off)),
    })
    out["agg_err"] = out["pred_agg_W"] - out["actual_W"]
    out["agg_err_off"] = out["pred_agg_off_W"] - out["actual_W"]
    out = out.sort_values("actual_W", ascending=False)

    print(f"\n=== Predicted win totals across {len(out)} teams (sorted by actual) ===")
    print(out.to_string(index=False))
    mae_real = (out["pred_real_W"] - out["actual_W"]).abs().mean()
    mae_agg = out["agg_err"].abs().mean()
    mae_agg_off = out["agg_err_off"].abs().mean()
    print(f"\nMAE in wins vs actual  ->  model on real stats: {mae_real:.1f}   "
          f"model on aggregated (conservation ON): {mae_agg:.1f}   "
          f"model on aggregated (conservation OFF): {mae_agg_off:.1f}")

    # --- 5-man lineup validation: ground truth for the roster shape (5-15,
    # --- most often 5) this program is actually asked to score, vs. the
    # --- 15-man team-rotation shape validated above. ---
    lineup_mape, lineup_summary = validate_against_lineups(feature_cols)
    print(f"\n=== 5-man lineup validation (POSS floor >= {LINEUP_POSS_FLOOR}) ===")
    print(f"Lineups considered: {lineup_summary['lineups_considered']}  |  "
          f"used: {lineup_summary['lineups_used']}  |  "
          f"skipped (missing player-season): {lineup_summary['skipped_missing_player_season']}  |  "
          f"skipped (traded/team mismatch): {lineup_summary['skipped_traded_team_mismatch']}  |  "
          f"skipped (ineligible player): {lineup_summary['skipped_ineligible_player']}  |  "
          f"skipped (zero minutes): {lineup_summary['skipped_zero_minutes']}")
    print("\nPOSS-weighted mean abs % error per feature, both rescaled to a common\n"
          "48-minute basis before comparing (5-man, asc) -- see validate_against_lineups():")
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(lineup_mape.to_string())
    print(f"\nOverall POSS-weighted mean abs % error (5-man): {lineup_mape.mean():.2f}%")

    # --- Where 15-man (team-rotation) and 5-man (this program's actual use
    # --- case) validation disagree: a feature that aggregates well at 15
    # --- players but badly at 5 matters more, since 5 is the shape the
    # --- program is most often asked about. The 15-man validation doesn't
    # --- surface the minutes-basis mismatch above because a real team's
    # --- top-15 by minutes already sums to ~240 total minutes on its own
    # --- (usage_scale ~= 1.0), so the rescale is nearly a no-op there. ---
    shape_cmp = pd.DataFrame({"mape_15man": mape, "mape_5man": lineup_mape})
    shape_cmp["diff_5_minus_15"] = shape_cmp["mape_5man"] - shape_cmp["mape_15man"]
    print("\n=== 15-man vs. 5-man validation: biggest disagreements (desc |diff|) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(shape_cmp.reindex(shape_cmp["diff_5_minus_15"].abs().sort_values(ascending=False).index).head(10).to_string())
    print(
        "Positive diff_5_minus_15 = the feature aggregates *worse* for a 5-man\n"
        "unit than for a 15-man rotation -- the case CLAUDE.md flags as mattering\n"
        "more, since 5 is the roster size this program is actually asked about."
    )

    # --- Star-stack stress test: 5 real, high-usage, ball-dominant players on ---
    # --- a fictional roster together. Expect poss_raw well above poss_target, ---
    # --- so usage_scale should come out clearly below 1. The guard is fit on ---
    # --- every real team-season (not just the training ones): it's answering ---
    # --- "has the league ever looked like this", for which every observation ---
    # --- counts, not "is this row out of sample". ---
    guard = fit_extrapolation_guard(teams_all[feature_cols], teams_all[TEAM_TARGET])
    demo_star_stack(qualified, feature_cols, model, guard)


if __name__ == "__main__":
    main()