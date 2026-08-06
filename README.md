# NBA Record Prediction Program

Predicts an NBA team's regular-season **win rate** from team-level statistics
using machine learning, and — via a player-to-team aggregation layer — predicts
the record a **fictional 5–15 player roster** (present or historic players) would
post in today's NBA.

## Current status

The project is built in two halves. The first is validated; the second is
working end-to-end but rests on estimates that can only be graded indirectly.

**Built & validated — the real-team model.**
Predicts a real NBA team's season win rate from its per-game stats, evaluated on
seasons the model never trained on (chronological holdout). Test MAE ≈ 2.6 wins.

**Built & integrated — the fictional-roster path.**
A user builds a roster interactively (by player *and* season), the roster is
aggregated into a single team stat-line, and the same ridge model scores it.
The aggregation includes **possession conservation** (the ball is finite, so five
players' summed shot volume is rescaled to a realistic possession count) and
**drops DEF_RATING** by design (no honest analogue exists for a team that never
played — see Limitations). This path has no ground truth: fictional teams never
played, so it's validated *indirectly* by aggregating real teams from their own
players and checking the result against the real team's line (~7.6 wins MAE).

**Built but NOT integrated — the usage→efficiency adjustment.**
A model of how a player's shooting efficiency responds to a change in usage rate
was built and tested, but found too weak to trust (see Test log). It exists as a
standalone script and is deliberately **not** wired into the aggregation.

**Resolved by removal — DEF_RATING and the gradient-boosting model.**
Both were tested thoroughly and dropped; the reasoning is preserved below so it
isn't re-litigated later.

## Project structure

| File | Purpose |
|------|---------|
| `pull_team_stats.py` | Pulls Base + Advanced team stats, merges on TEAM_ID + SEASON, converts counting stats to per-game, writes `team_season_stats.csv`. Data only. |
| `pull_player_stats.py` | Pulls current-season per-player Base + Advanced stats → `player_season_stats.csv`. Data only. |
| `pull_player_seasons_all.py` | Pulls per-player stats for every season 1996-97 onward (the roster-input pool) → `player_all_seasons.csv`. Data only. |
| `pull_lineups.py` | Pulls 5-man lineup stats (Base + Advanced), 2014-15 onward → `lineup_season_stats.csv`. Ground truth for `aggregate_team.py`'s 5-man validation. Data only. |
| `train_model.py` | Loads the team CSV, splits by season, trains the ridge win-rate model, prints metrics and standardized coefficients (the model's top drivers). |
| `aggregate_team.py` | Aggregates 5–15 players into one team stat-line (possession conservation, DEF_RATING dropped), and self-validates by rebuilding every real team from its own roster. |
| `predict_fictional_roster.py` | Interactive roster builder (by player + season); aggregates and scores the roster with the ridge model. |
| `usage_efficiency_model.py` | **Standalone, not integrated.** Fits and tests the usage→efficiency (USG%→TS%) response model. Kept as a documented experiment. |
| `SPEC.md` | Full project specification. |
| `CLAUDE.md` | Working agreements, incl. the aggregation contract. |

## Setup & commands

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python pull_team_stats.py            # team data     -> team_season_stats.csv
python pull_player_seasons_all.py    # player pool    -> player_all_seasons.csv
python train_model.py                # train + report the real-team model
python aggregate_team.py             # aggregation + real-team validation
python predict_fictional_roster.py   # build a roster and predict its record
```

## Configuration

Knobs live at the top of each script:

- `NUM_SEASONS` — how many seasons of team data to pull (`pull_team_stats.py`)
- `N_TEST_SEASONS` — recent seasons held out for testing (`train_model.py`)
- `ALPHAS` — ridge penalty grid searched by cross-validation
- `MIN_GP`, `MIN_MPG` — player eligibility thresholds (`aggregate_team.py`)

## Methodology

**Target — win rate, not raw wins.** The target is `W_PCT`. Raw win totals aren't
comparable across seasons of different length (e.g. the shortened 2019-20 and
2020-21 seasons), so win *rate* is used and multiplied by 82 to report a
projected record.

**Features — per-game and rate stats.** Season-total counting stats (points,
rebounds, etc.) are divided by games played so a 72-game and an 82-game season
sit on the same scale. Stats already expressed as rates (shooting percentages,
efficiency ratings, pace) are left as-is.

**Leakage control.** Any column that directly contributes to the target is
removed, along with redundant columns: wins/losses and their ranks, `W_PCT`-derived
fields, `PLUS_MINUS`, and `NET_RATING`. These would let the model reconstruct the
target instead of learning from basketball stats. `FT_PCT` and similar are removed
because they're recoverable from makes and attempts, which are kept.

**Evaluation — chronological holdout.** The model trains on older seasons and is
tested on the most recent seasons it has never seen. This mirrors real forecasting
and avoids the mild leakage of a random split, where a team's adjacent seasons
could straddle the train/test line.

**Aggregation — from players to a team line.** Box-score events are summed
across the roster; possession-linked volume is then rescaled so total possessions
match a pace anchor derived from the players' own PACE (see Test log). Rates
(OFF_RATING, EFG_PCT, TM_TOV_PCT, ...) are recomputed from the aggregated totals
so they stay internally consistent. The aggregation contract (preconditions,
postconditions, invariants) lives in `CLAUDE.md`.

## Results — real-team model

**Production model:** ridge regression trained on the modern era (2014-15 onward).

- Held-out-season win rate: **MAE ≈ 0.032 → about 2.6 wins** over 82 games.
- **Test R² ≈ 0.94**, with essentially no train/test gap (no overfitting).
- **Dominant drivers:** offensive and defensive efficiency. Note this describes
  the *real-team* model, which uses real DEF_RATING; the fictional-roster path
  deliberately omits DEF_RATING (see Limitations), which is the single largest
  source of its wider error floor.

## Test log

The decisions below were made from experiments, not intuition. Each is recorded
as *what was tested → what the evidence showed → what was decided*, so the
reasoning survives and isn't rebuilt from scratch.

**Ridge vs. gradient boosting (real-team model).** Compared at several dataset
sizes. Gradient boosting overfit heavily on small data and never overtook ridge:

| Training rows | GB train R² | GB test R² | GB gap |
|---------------|-------------|------------|--------|
| 300 (10 seasons) | 0.986 | 0.916 | 0.070 |
| 360 (12 seasons) | 0.981 | 0.919 | 0.062 |
| 540 (18 seasons) | 0.973 | 0.931 | 0.042 |

*Decision:* ridge is the production model. Season-level prediction is inherently
small-data (~30 team-seasons per year), which favors a simple regularized linear
model. Gradient boosting is not shipped.

**TM_TOV_PCT formula bug.** The aggregated turnover rate used a plays-based
denominator (`FGA + 0.44·FTA + TOV`) while the team CSV stores a possession-based
one (`TOV / POSS`), biasing it low on every team. Because it's a high-coefficient
feature, the error inflated predicted wins across the board. *Decision:* fixed the
denominator; aggregated win-MAE dropped from **14.3 → 6.5**. Established the
standing rule that a derived rate must match the CSV column's exact definition,
not just its units.

**DEF_RATING sub-model.** Tested whether team DEF_RATING could be predicted from
aggregatable defensive box-score stats (steals, blocks, defensive rebounds,
fouls, pace). *Evidence:* test R² ≈ 0.19 — the box score doesn't carry team
defense. Swapping the real DEF_RATING for this prediction was no better than
dropping the column entirely. *Decision:* not used.

**DEF_RATING personal composite.** A single feature built by regressing personal
defensive stats against real DEF_RATING, replacing both DEF_RATING and its raw
inputs. *Evidence:* it barely beat dropping the column on real-team *training*
(~7.57 vs. ~7.60 wins MAE), and that edge traced to collapsing four collinear
columns into one (a mild regularization effect) rather than to recovered defensive
signal; it vanished through aggregation. *Decision:* not used.

**DEF_RATING awards composite.** A rule-based score from DPOY / All-Defensive
selections plus box-score defensive character. *Evidence:* equal to or worse than
simply dropping DEF_RATING. *Decision:* not used.

**DEF_RATING — final.** All synthesis routes failed for the same structural
reason: DEF_RATING is opponent-dependent and a fictional roster has no honest way
to produce it. *Decision:* **dropped** from the fictional-roster path. The
resulting ~7.6-win aggregated MAE is accepted as an honest floor rather than
papered over. (The real-team model in `train_model.py` still uses real
DEF_RATING; only the fictional path drops it.)

**Possession conservation.** Summing five players' box-score volume can imply more
possessions than a real game contains. *Approach:* anchor the roster to a
possession target = minutes-weighted average of the players' own PACE (validated:
tracks real team PACE at ~1.4% mean abs error, corr 0.985), then scale volume so
possessions match it. *Evidence:* possession-denominated rates (OFF_RATING,
TM_TOV_PCT) are provably invariant to the scaling (ON vs OFF differ by ~1e-14);
`usage_scale` is ≈ 1.0 on real teams (their players already fit together) and
**0.56 on a five-star stress roster** (where the ball genuinely can't stretch to
fit five ball-dominant scorers). *Decision:* **integrated.** Near-no-op on real
teams by design; the mechanism only bites where it should.

**Usage→efficiency response.** Tested whether a player's shooting efficiency
(TS%) can be adjusted for a change in usage rate, using real "situation-change"
season pairs (player traded, or a high-usage teammate arrived/left) to isolate
externally-caused usage shifts. Split by an on-ball/off-ball index (tracking-stat
derived). *Evidence:* the effect is **real but weak, and only for off-ball
players** — off-ball slope significant (usage down → efficiency up, p ≈ 0.005) but
low-signal (R² ≈ 0.08); on-ball slope indistinguishable from zero (p ≈ 0.61).
A continuous-slope variant did not beat the two-bucket split. Per-player
predictions are near coin-flips; the model's value, if any, is only at the roster
average. *Decision:* **not integrated.** The signal is too weak to justify wiring
into the prediction, and any five-star roster forces usage swings far beyond the
observed data range (predictions there run entirely on the extrapolation taper).
Kept as a documented experiment in `player_efficiency.py`.

**Perturbation test replacing MAPE×coefficient error-impact ranking.** The old
`error_impact()` (crossing per-feature aggregation MAPE with the ridge model's
standardized coefficients) is unreliable under collinearity: PACE and POSS
correlate at ~0.99 with large, opposite-signed coefficients (PACE ≈ -0.29,
POSS ≈ +0.13), so a coefficient's raw magnitude overstates how much *that
feature's own* error moves a prediction. *Approach:* `perturbation_impact()`
instead perturbs each feature by its own measured MAPE (both directions),
re-predicts with the fitted model, and averages the resulting win-count
change over all 30 teams. *Evidence:* the two rankings disagree most exactly
where predicted — PACE and DREB_PCT swap rank by 7-9 places between the two
methods. *Decision:* **perturbation_impact() is now primary**; the old
function is kept as `error_impact_deprecated()` so both print side by side.
Known limitation of the new diagnostic too: it perturbs one feature at a
time, which assumes independent feature errors — false here, since PACE,
POSS, FGA, and PTS errors all originate in the same usage-scaling step.

**5-man lineup validation (vs. the existing 15-man team validation).** The
15-man validation (`aggregate_team.py` rebuilding a real team from its own
top-15-by-minutes roster) is the wrong shape to trust: this program is
actually asked to score 5-15 players, most often 5, not a full team
rotation. *Approach:* `pull_lineups.py` pulls every real 5-man lineup's own
observed Base + Advanced line (`lineup_season_stats.csv`, 2014-15 onward,
23,470 lineup-seasons); for each lineup clearing a 250-total-possession floor
(chosen from the pulled data's own distribution — median lineup-season is
only ~45 possessions, 250 sits around the 95th percentile, roughly each
team's handful of most-used units per season), its five players' own season
stats are run through `aggregate_team()` and compared to the lineup's real
line, weighted by possessions (827 of 1,005 qualifying lineups used; 178
skipped for the traded-player caveat — a player's `player_all_seasons.csv`
row reflects only their final team of the season, which can differ from the
lineup's team). *First pass, naive comparison:* rate-like features
(OFF_RATING, EFG_PCT, AST_TO, AST_RATIO, TM_TOV_PCT, OREB_PCT, DREB_PCT, PACE)
validated fine (single digits to ~20% weighted MAPE), but every additive
box-score total, plus POSS, ballooned to 300-460% weighted MAPE. Root cause:
`aggregate_team()` rescales a roster to `TOTAL_TEAM_MINUTES` (240 = a full,
uninterrupted 48-minute game — SPEC.md's deliberate premise that a fictional
roster plays the whole game without substitution), while a real 5-man
lineup's own recorded line covers only the ~15-20 partial minutes/game that
exact five actually shared the floor before a substitution — two bases that
aren't commensurable for anything that scales with minutes played. The
15-man validation never surfaced this because a real team's top-15-by-minutes
roster already sums to ~240 combined minutes on its own (`usage_scale` ≈ 1.0
there), making the rescale nearly a no-op — it only bites at the roster size
(5) this program actually predicts. *Fix:* rather than change
`aggregate_team()`'s total-minutes assumption (which would fix the box-score
totals but not POSS/PACE — `aggregate_team()` derives POSS from PACE, already
a per-48-minutes quantity independent of total minutes), `validate_against_lineups()`
rescales the real lineup's own line up to a 48-minute-equivalent basis
(`48 / row["MIN"]`) before comparing — extrapolating "what this lineup's own
box score would look like over a full 48 minutes at its own on-court rate,"
the same quantity `aggregate_team()` is built to produce. Rate/percentage
columns are left as-is (already floor-time-invariant). *Evidence after the
fix:* overall weighted MAPE dropped from **276.70% to 10.33%**; POSS now
tracks PACE closely (1.81% vs. 1.72%), confirming both are on the same basis.
FTM, FTA, BLKA, PFD, and BLK still validate meaningfully worse at 5-man than
15-man (~18-24% vs. ~3-6%) — these are exactly the lower-frequency events
(free throws, blocks) where a 5-player sample is noisiest, a real remaining
signal rather than a units artifact. *Decision:* **rescale integrated** into
`validate_against_lineups()`; see Limitations for what the residual 5-man-vs-
15-man gap in low-frequency stats means for a fictional prediction.

## Limitations & next steps

- **No ground truth for the end goal.** Fictional/cross-era teams never played,
  so their predictions are informed estimates that can't be directly verified.
  The pipeline is graded only *indirectly*, on real teams.
- **DEF_RATING is dropped on the fictional path.** The part of defense that lives
  in scheme and opponent behavior can't be manufactured from a roster's own
  stats by any method tested (see Test log). This is the largest single component
  of the ~7.6-win aggregated error floor.
- **"Prime" is a heuristic.** The roster tool's "Prime" season is the eligible
  season with the highest PIE — a box-score summary that is **not** era- or
  pace-adjusted. It's a reasonable no-extra-computation stand-in, not a
  considered "best season" judgment.
- **No era adjustment yet.** Cross-era rosters mix different styles of basketball
  (pace, three-point volume, rules). A historic player is currently scored on his
  raw stat line, unadjusted for the era he played in. This is the main planned
  build.
- **Usage/fit only partially modeled.** Possession conservation handles the
  *volume* side of stacking stars (the ball is finite). The *efficiency* side
  (how a star's shooting changes in a reduced role) was tested and found too weak
  to include, so a five-star roster's efficiency is currently taken at face value.
- **Small data.** ~30 team-seasons per year caps model complexity — the reason a
  simple regularized model is the right call.
- **Coaching / intangibles.** In the SPEC's scope but not built; they belong on
  top of a validated statistical core, not inside it.
- **5-man validation rests on a linear 48-minute extrapolation.** Comparing
  `aggregate_team()`'s output (a full, uninterrupted 48-minute game) to a real
  5-man lineup's own recorded line (only its actual ~15-20 partial minutes/game)
  requires putting both on the same basis — `validate_against_lineups()` does
  this by scaling the real lineup's line up by `48 / MIN` (see Test log). That
  extrapolation assumes the lineup's own on-court rate would hold linearly
  across a full game; a real lineup's rate reflects fatigue, opponent
  adjustments, and matchup-specific chemistry over its actual (short) run,
  none of which necessarily holds at 4x-plus the minutes. With that rescale in
  place, overall weighted MAPE is ~10%, and low-frequency box-score events
  (FTM, FTA, BLKA, PFD, BLK — free throws and blocks) still validate
  meaningfully worse at 5-man (~18-24%) than 15-man (~3-6%), consistent with a
  smaller sample amplifying noise in rarer events. Read a fictional 5-player
  roster's predicted free-throw and block volume with more caution than its
  shooting/scoring/rebounding totals.
- **Perturbation-test limitation.** `perturbation_impact()` (see Test log)
  perturbs one feature at a time, assuming independent aggregation errors.
  That's false here — PACE, POSS, FGA, and PTS errors all originate in the
  same usage-scaling step, so their real, combined error moves together, not
  independently, and this diagnostic can't capture that correlation.