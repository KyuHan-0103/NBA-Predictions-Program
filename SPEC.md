# NBA Record Prediction — Specification

## Overview

Inspired by the "82-0" roster-building games, this program predicts a fictional
NBA team's success by projecting its win-loss record. It takes 5–15 real players
(present or historic), aggregates them into a single team statistical profile,
and scores that profile with a model trained on real NBA teams. It keeps the
spirit of fantasy — players cross eras without injury or aging — while grounding
the prediction in real statistical relationships.

The longer-term ambition is to incorporate style-of-play and fit (how players
operate together and under different systems). The statistical core is built and
validated first; those softer factors sit on top of it and are not yet built.

## Goals

- Predict a fictional team's win-loss record **in today's NBA** (compared only
  against current teams).
- Accept **5–15 players**, present or historic, and return a projected record
  over an 82-game season.
- **Non-goals:** directly comparing two fictional teams; modeling injuries,
  fatigue, or aging. A 5-man roster is not penalized for lacking depth — minutes
  are scaled up to a full team's allotment.
- **Coaching (planned, not built):** optionally choosing a coach who adjusts how
  fully players are utilized.

## Architecture

The pipeline runs in four stages:

1. **Data (`pull_*` scripts).** Pull and format Base + Advanced stats from the
   NBA Stats API (`nba_api`): team stats per season, and player stats per season
   back to 1996-97 (the first season Advanced player data exists).
2. **Real-team model (`train_model.py`).** A ridge regression predicts team
   `W_PCT` from all 27 team features, evaluated on a chronological holdout. This
   is the scoring model the fictional path also reuses — a smaller feature set
   for that path was tested and rejected (README Test log). It fits the
   **logit** of `W_PCT` (`ln(p/(1-p))`, inverted with `1/(1+e^-z)`) so a
   predicted win rate is bounded in `(0, 1)` for any roster, however extreme —
   the untransformed model returned win rates above 1 and below 0 on stacked
   rosters. Callers still hand in and read plain `W_PCT`. Ships an
   **extrapolation guard** alongside, because a bounded prediction no longer
   reveals when it is unsupported, and the **prediction interval** every
   predicted record is reported with.
3. **Aggregation (`aggregate_team.py`).** Convert 5–15 players into one
   team-shaped stat-line that the ridge model can score. Includes possession
   conservation; drops DEF_RATING. Self-validates against real teams.
4. **Roster input & prediction (`predict_fictional_roster.py`).** An interactive
   tool builds a roster, enforces the aggregation contract's preconditions at
   input time, aggregates, and prints the predicted record.

### Aggregation (aggregate_team.py)

Combines a roster into the exact feature shape the ridge model consumes. Feature
families are handled by how they actually combine:

- **Additive box-score events** (points, rebounds, assists, ...): summed across
  the roster after rescaling each player's minutes so the roster totals a full
  team's minutes.
- **Possession conservation:** the ball is finite, so summed player volume can
  imply more possessions than a game contains. Volume is rescaled so total
  possessions match a pace anchor — the minutes-weighted average of the players'
  own PACE (validated to track real team PACE closely). Rates are recomputed from
  the rescaled totals and are provably unchanged by the rescale.
- **Derived rates** (OFF_RATING, EFG_PCT, AST_TO, AST_RATIO, TM_TOV_PCT):
  recomputed from the aggregated totals with standard formulas, each matching the
  team CSV's exact column definition.
- **Opponent-dependent rates** (OREB_PCT, DREB_PCT): approximated from the
  players' own rebound shares — the softest estimates in the pipeline, flagged as
  low-confidence in the output, and **clipped** into `[0, 1]` (five bigs claim
  ~148% of the opponent's misses otherwise) with every clip recorded in the
  output's `attrs`. Any other `*_PCT` leaving `[0, 1]` raises instead of being
  clipped: it is recomputed from the roster's own totals, so out of range means
  a bug, not an approximation.
- **DEF_RATING:** dropped. It is opponent-dependent and has no honest analogue for
  a team that never played (see README Test log for the alternatives tested).

The full aggregation **contract** — preconditions, postconditions, and invariants
— lives in `CLAUDE.md` and is the authority for what a valid aggregation must
guarantee.

### Roster input (predict_fictional_roster.py)

Backed by the full player pool from `pull_player_seasons_all.py` (1996-97 onward).
The user builds a roster interactively, one player at a time, by name.

- Each pick is a specific **(player, season)** pair, not just a person — the same
  player from two seasons is two separately selectable entries (2019-20 Curry ≠
  2022-23 Curry).
- For each player the user chooses either:
  - **"Prime":** the eligible season with the highest **PIE** (Player Impact
    Estimate — an NBA-computed, box-score-derived summary). A heuristic, not an
    era/pace-adjusted "best season" judgment.
  - **A specific season**, entered as e.g. `2019-20`.
- **Eligibility** reuses the aggregation filter (GP ≥ `MIN_GP`, MIN ≥ `MIN_MPG`)
  plus a NaN/inf/all-zero guard, applied at input time so the aggregation
  contract's precondition is already satisfied before a roster reaches
  `aggregate_team()`.
- Name matching is case/diacritic/punctuation-insensitive ("doncic" finds
  "Dončić"; multiple matches prompt for disambiguation).

**Rejection cases (re-prompt, never silently drop or substitute):** player not
found; malformed or out-of-range season label; player didn't play the named
season; the (player, season) row fails eligibility or has unusable data; the same
(player, season) picked twice; a name matches multiple players without
disambiguation. The tool also blocks finishing with fewer than 5 players and stops
accepting picks at 15.

## Modeling scope — built vs. tested-and-excluded

- **Built & integrated:** the real-team ridge model; the logit target and its
  extrapolation guard; player-to-team aggregation; possession conservation; the
  `*_PCT` clip and its postcondition check; the 90% prediction interval; the
  stress rosters and the printed expected-vs-fitted coefficient signs.
- **Tested & deliberately excluded:** every DEF_RATING synthesis approach
  (sub-model, personal composite, awards composite), the gradient-boosting
  model, and a reduced rates-and-defence feature set for the fictional path
  (fixed the coefficient signs, cost 7.7 → 12.9 wins MAE on aggregated rosters)
  — all dropped with evidence (see README Test log).
- **Built but NOT integrated:** the usage→efficiency (USG%→TS%) response model.
  Real but weak signal (off-ball only, R² ≈ 0.08; on-ball null), too unreliable to
  wire into predictions. Retained as a standalone experiment in
  `player_efficiency.py`.
- **Built but NOT integrated:** the 5-man lineup DEF_RATING model
  (`lineup_defense_model.py`) — predicts a lineup's DEF_RATING from its players'
  *prior-season* individual stats only, with all on-court team context forbidden,
  so it is non-circular by construction (unlike the three excluded approaches
  below). Real but small signal under rolling-origin CV: season-centered
  lineup-level R² ≈ 0.06 (95% CI includes zero) against a label-noise ceiling of
  0.33, and a team-level calibration of `DRtg_used = 37.85 + 0.679 × DRtg_est`
  (R² ≈ 0.23) that still beats predicting the league average by only ~4% of
  team-level MAE. Retained as a standalone experiment; DEF_RATING stays dropped
  from the aggregation.
- **Planned, not built:** coaching adjustment; era adjustment for cross-era
  rosters; style/fit intangibles.

## Constraints & assumptions

- Roster size: **5–15 players.**
- Players selectable from **1996-97 onward** (first season with Advanced player
  data; earlier seasons return zero rows, not garbage).
- A player-season qualifies only if it clears the eligibility filter (GP ≥
  `MIN_GP`, MIN ≥ `MIN_MPG`) and its required advanced columns are finite and not
  all zero — the same precondition the aggregation contract requires.
- No injuries, fatigue, or aging are modeled; a thin roster is not penalized for
  depth.
- **"Prime" = highest PIE among eligible seasons** — a heuristic, not a subjective
  or era-adjusted best-season judgment.
- Fictional predictions have **no ground truth** and can only be graded indirectly,
  by validating the aggregation on real teams.
- A predicted record is always **possible** (the logit target bounds it) but that
  is not evidence it is **supported**. Aggregated rosters land outside the region
  of feature space the model was fit on, so a fictional prediction must be read
  together with the extrapolation guard's ratio — see README Limitations.
- A predicted record is reported with a **90% interval** covering model error and
  82-game outcome randomness, and with the two terms it does *not* cover named:
  the dropped DEF_RATING (~2.6 wins) and the roster's extrapolation ratio.
  Neither is quantifiable from available data, so neither is folded in.