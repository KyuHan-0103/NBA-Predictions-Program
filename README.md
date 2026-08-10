# NBA Record Prediction Program

Predicts an NBA team's regular-season **win rate** from team-level statistics
using machine learning, and — via a player-to-team aggregation layer — predicts
the record a **fictional 5–15 player roster** (present or historic players) would
post in today's NBA.

## Current status

> **Verified end to end, final state (19-check suite, all passing).** The
> fictional path is back on the same 27 features `train_model.py` fits, minus
> `DEF_RATING`. Confirmed: holdout MAE **4.85 wins / R² 0.813**; aggregated-roster
> MAE **7.6 wins** over all 360 team-seasons; guard `edge` 5.78 / `limit` 6.36 with
> median ratio 1.70x and the flag firing on 100% of aggregated rosters; held-out
> residual SD 0.270 log-odds; the `*_PCT` clip fires on the five-centre roster
> (`DREB_PCT` 1.482 → 1.000) and `_check_pct_range` raises on a forced
> `TM_TOV_PCT` of 1.4; all four `prediction_interval` reference rows reproduce
> exactly; saturation is flagged rather than printed as a range; and
> `coefficient_signs()` reports 7 of 16 priced coefficients fitting backwards.
> `lineup_defense_model.py` and `usage_efficiency_model.py` live in separate
> folders and are outside this suite.


The project is built in two halves. The first is validated; the second is
working end-to-end but rests on estimates that can only be graded indirectly.

**Built & validated — the real-team model.**
Predicts a real NBA team's season win rate from its per-game stats, evaluated on
seasons the model never trained on (chronological holdout). Test MAE ≈ 2.6 wins.

**Built & integrated — the fictional-roster path.**
A user builds a roster interactively (by player *and* season), the roster is
aggregated into a single team stat-line, and a ridge model scores it. The
aggregation includes **possession conservation** (the ball is finite, so five
players' summed shot volume is rescaled to a realistic possession count) and
**drops DEF_RATING** by design (no honest analogue exists for a team that never
played — see Limitations). This path has no ground truth: fictional teams never
played, so it's validated *indirectly* by aggregating real teams from their own
players and checking the result against the real team's line.

It is scored by the same 27-feature model, on purpose: a smaller feature set
built to make the coefficients read as basketball was tested and rejected for
costing accuracy. Every prediction is reported with
a **90% interval** whose two components are separated, and with an explicit
note of the two terms that interval does *not* cover.

**Built but NOT integrated — the usage→efficiency adjustment.**
A model of how a player's shooting efficiency responds to a change in usage rate
was built and tested, but found too weak to trust (see Test log). It exists as a
standalone script and is deliberately **not** wired into the aggregation.

**Built but NOT integrated — the 5-man lineup DEF_RATING model.**
A fourth attempt at the DEF_RATING problem, under a construction that makes the
circularity of the earlier three *causally impossible*: a lineup's DEF_RATING is
predicted from its five players' **prior-season** individual stats only, with all
on-court team context (player DEF/OFF/NET_RATING, PLUS_MINUS, on/off splits)
forbidden from every season. Evaluated by rolling-origin CV over six held-out
seasons, it finds **real but small** signal: season-centered lineup-level
R² **0.058 [−0.006, +0.122]** against a label-noise ceiling of 0.330, and at
team level a stable calibration of `DRtg_used = 37.85 + 0.679 × DRtg_est`
(R² **0.233**, 133 held-out team-seasons) — which still only beats "predict the
league average" by 4% of team-level MAE. Lives in `lineup_defense_model.py`;
deliberately **not** wired into the aggregation (see Test log and Limitations).

**Resolved by removal — DEF_RATING and the gradient-boosting model.**
Both were tested thoroughly and dropped; the reasoning is preserved below so it
isn't re-litigated later.

## Project structure

The main pipeline sits at the repo root. The two experiments that are documented
but **not integrated** live in their own folders, alongside the data pulls that
feed only them, so the root contains nothing the shipped pipeline doesn't use.

```
aggregate_team.py  train_model.py  perturbation_tests.py
predict_fictional_roster.py
pull_team_stats.py  pull_player_seasons_all.py  pull_lineups.py
*.csv  README.md  SPEC.md  CLAUDE.md  AGGREGATION_CONTRACT.md  requirements.txt

def_rating_testing/   lineup_defense_model.py, pull_player_defense.py
usg_testing/          usage_efficiency_model.py, pull_player_history.py
docs/                 pull_player_stats.py + the GP/MIN histograms it produces
```

| File | Purpose |
|------|---------|
| `pull_team_stats.py` | Pulls Base + Advanced team stats, merges on TEAM_ID + SEASON, converts counting stats to per-game, writes `team_season_stats.csv`. Data only. |
| `docs/pull_player_stats.py` | **Exploratory, not in the pipeline.** Pulls current-season per-player stats → `player_season_stats.csv` and plots the GP / MIN histograms in `docs/` that `MIN_GP` and `MIN_MPG` were chosen from. Nothing else reads its CSV. Data only. |
| `pull_player_seasons_all.py` | Pulls per-player stats for every season 1996-97 onward (the roster-input pool) → `player_all_seasons.csv`. Data only. |
| `pull_lineups.py` | Pulls 5-man lineup stats (Base + Advanced), 2014-15 onward → `lineup_season_stats.csv`. Ground truth for `aggregate_team.py`'s 5-man validation. Data only. |
| `def_rating_testing/pull_player_defense.py` | Pulls per-player *individual* defensive descriptors — bio (height/weight/age), closest-defender tracking (opponent FG% on shots he defended, 2013-14 onward), hustle (2016-17 onward) → `player_defense_stats.csv`. Data only. |
| `train_model.py` | Loads the team CSV, splits by season, trains the ridge win-rate model on `logit(W_PCT)`, prints metrics and standardized coefficients (the model's top drivers). Also home to `logit`/`inv_logit`, `ridge_step()` (reaches the `RidgeCV` inside the target-transform wrapper), the extrapolation guard every roster prediction is reported against, and `prediction_interval()` / `interval_report()`. |
| `aggregate_team.py` | Aggregates 5–15 players into one team stat-line (possession conservation, DEF_RATING dropped, `*_PCT` clipped to the contract), and self-validates by rebuilding every real team from its own roster. Also home to `coefficient_signs()` (expected vs. fitted signs, printed) and the stress rosters. |
| `perturbation_tests.py` | The aggregation's per-feature error-impact diagnostics (`perturbation_impact()` + the deprecated MAPE × coefficient ranking), split out of `aggregate_team.py`. Runs standalone, and `aggregate_team.py` still prints the identical tests via `run_perturbation_tests()`. |
| `predict_fictional_roster.py` | Interactive roster builder (by player + season); aggregates and scores the roster with the ridge model. |
| `usg_testing/pull_player_history.py` | Pulls player Advanced + tracking stats (2013-14 onward, the SportVU floor) → `player_history_stats.csv`. Feeds `usage_efficiency_model.py` only; nothing in the main pipeline reads it. Data only. |
| `usg_testing/usage_efficiency_model.py` | **Standalone, not integrated.** Fits and tests the usage→efficiency (USG%→TS%) response model. Kept as a documented experiment. |
| `def_rating_testing/lineup_defense_model.py` | **Standalone, not integrated.** Estimates a 5-man lineup's DEF_RATING from its players' prior-season individual stats (permutation-invariant pooling, possession-weighted, rolling-origin CV by season), with the possession-floor sweep, label-noise ceiling, both baselines, and the reliability / team-change diagnostics. Kept as a documented experiment. |
| `SPEC.md` | Full project specification. |
| `CLAUDE.md` | Working agreements, incl. the aggregation contract. |

## Setup & commands

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# The scripts in def_rating_testing/, usg_testing/ and docs/ read and write CSVs
# relative to the working directory, so run them from the repo root as below.

python pull_team_stats.py            # team data      -> team_season_stats.csv
python pull_player_seasons_all.py    # player pool    -> player_all_seasons.csv
python pull_lineups.py               # 5 man lineups  -> lineup_season_stats.csv
python def_rating_testing/pull_player_defense.py   # player defense -> player_defense_stats.csv
python train_model.py                # train + report the real-team model
python aggregate_team.py             # aggregation + real-team validation
python perturbation_tests.py         # aggregation error-impact diagnostics only
python predict_fictional_roster.py   # build a roster and predict its record
python def_rating_testing/lineup_defense_model.py  # 5-man lineup DEF_RATING experiment
python usg_testing/pull_player_history.py         # tracking data for the usage experiment
python usg_testing/usage_efficiency_model.py      # usage -> efficiency experiment
python docs/pull_player_stats.py                  # optional: regenerate the GP/MIN histograms
```

## Configuration

Knobs live at the top of each script:

- `NUM_SEASONS` — how many seasons of team data to pull (`pull_team_stats.py`)
- `N_TEST_SEASONS` — recent seasons held out for testing (`train_model.py`)
- `ALPHAS` — ridge penalty grid searched by cross-validation
- `MIN_GP`, `MIN_MPG` — player eligibility thresholds, chosen from
  `docs/2025-26 GP Histogram Distribution.png` and
  `docs/2025-26 MIN Histogram Distribution.png` (`aggregate_team.py`)
- `POSS_FLOORS`, `PRIMARY_POSS_FLOOR` — lineup possession floors compared, and the one chosen from that comparison (`lineup_defense_model.py`)
- `MIN_TRAIN_SEASONS`, `LINEUP_ALPHAS` — first rolling-origin fold's training depth, and the ridge grid (deliberately wider than `train_model.py`'s — see Test log) (`lineup_defense_model.py`)

## Reproducing the reported results

Every number in this README comes from the committed CSVs, not from a live API
call. **This matters more than the config constants below:** the NBA Stats API
returns live data, so re-running `pull_team_stats.py` in a later season pulls a
different 12 seasons and every reported figure shifts. The CSVs are committed for
exactly this reason. Re-pull only when you intend the numbers to move.

**Data snapshot the reported numbers come from**

| File | Rows | Cols | Seasons |
|---|---:|---:|---|
| `team_season_stats.csv` | 360 | 33 | 2014-15 .. 2025-26 (12) |
| `player_all_seasons.csv` | 14,569 | 29 | 1996-97 .. 2025-26 (30) |
| `lineup_season_stats.csv` | 23,470 | 38 | 2014-15 .. 2025-26 (12) |
| `player_defense_stats.csv` | 6,896 | 19 | 2013-14 .. 2025-26 (13) |
| `player_season_stats.csv` | 582 | 42 | 2025-26 only |

Note that `aggregate_team.py`'s and `perturbation_tests.py`'s own validation
sections pull the **current** season's player stats live rather than from a CSV,
so those two are the exception — their output moves as the season progresses.

**Configuration that produces the reported figures**

| Constant | Value | File | Controls |
|---|---|---|---|
| `NUM_SEASONS` | 12 | `pull_team_stats.py` | the 360 team-seasons everything is fit on |
| `START_SEASON` | 1996-97 | `pull_player_seasons_all.py` | the roster-input pool |
| `START_SEASON` | 2014-15 | `pull_lineups.py` | matches the team CSV's range |
| `GROUP_QUANTITY` | 5 | `pull_lineups.py` | 5-man lineups |
| `N_TEST_SEASONS` | 2 | `train_model.py` | the chronological holdout → **4.85 W / R² 0.813** |
| `ALPHAS` | `logspace(-3, 3, 25)` | `train_model.py` | ridge grid (selected α sits interior) |
| `LOGIT_EPS` | 1e-6 | `train_model.py` | poles clipped off `logit()`; never binds on real data |
| `EDGE_QUANTILE` | 0.99 | `train_model.py` | guard `edge` = **5.78** (`limit` = **6.36**, the max) |
| `MARGINAL_RATIO` | 1.15 | `train_model.py` | above this, the guard reports "unfamiliar", not "unsupported" |
| `INTERVAL_Z` | 1.645 | `train_model.py` | the 90% interval |
| `DEF_RATING_COST_WINS` | 2.3 | `train_model.py` | the uncovered term named next to every interval |
| `GAMES_PER_SEASON` | 82 | both | win-rate → record |
| `MIN_GP` | 5 | `aggregate_team.py` | eligibility (chosen from the histograms in `docs/`) |
| `MIN_MPG` | 6 | `aggregate_team.py` | eligibility |
| `TOTAL_TEAM_MINUTES` | 240.0 | `aggregate_team.py` | 5 x 48; the rescale the guard's displacement traces to |
| `LINEUP_POSS_FLOOR` | 250 | `aggregate_team.py` | 5-man validation → **827 of 1,005** lineups used |
| `PRIMARY_POSS_FLOOR` | 250 | `def_rating_testing/lineup_defense_model.py` | the defensive experiment's floor |
| `MIN_TRAIN_SEASONS` | 6 | `def_rating_testing/lineup_defense_model.py` | first rolling-origin fold → 6 held-out seasons |
| `LINEUP_ALPHAS` | `logspace(-3, 6, 40)` | `def_rating_testing/lineup_defense_model.py` | wider than `ALPHAS` on purpose (see Test log) |
| `CI_LEVEL` | 0.95 | `def_rating_testing/lineup_defense_model.py` | across-fold CIs, t with df = 5 |

**Environment.** `requirements.txt` bounds major versions rather than pinning
exactly, so the install doesn't rot. The reported figures were produced on:

```
pandas 3.0.2    numpy 2.4.4    scikit-learn 1.8.0    scipy 1.17.1
```

Record your own with `pip freeze > requirements-lock.txt` before quoting numbers,
since `RidgeCV`'s selected alpha and therefore the coefficients can move slightly
across scikit-learn versions.

**Expected output.** `python train_model.py` should print test R² 0.813, MAE
0.0591 (4.85 wins), guard `edge` 5.78 / `limit` 6.36, and a selected alpha
strictly inside the grid. If the alpha is pinned at either endpoint, that's a bug
report rather than a result — see the Test log.


## Methodology

**Target — win rate, not raw wins.** The target is `W_PCT`. Raw win totals aren't
comparable across seasons of different length (e.g. the shortened 2019-20 and
2020-21 seasons), so win *rate* is used and multiplied by 82 to report a
projected record.

**Target transform — logit, not raw `W_PCT`.** The model fits
`ln(p / (1-p))` and inverts with `1 / (1 + e^-z)`, so a prediction is in
`(0, 1)` for any input at all. A win rate is a proportion and cannot leave
`[0, 1]`; a ridge line does not know that, and the fictional path pushes it far
enough to matter — on the untransformed target the five-star stress roster
predicted a win rate of **1.004** with possession conservation on and **−2.609**
with it off. Callers are unaffected (`build_model()` still takes and returns
plain `W_PCT`; `TransformedTargetRegressor` applies the transform internally).
Because a bounded output can no longer announce that it is nonsense, the model
ships with an extrapolation guard next to it — see Test log.

**Features — per-game and rate stats.** Season-total counting stats (points,
rebounds, etc.) are divided by games played so a 72-game and an 82-game season
sit on the same scale. Stats already expressed as rates (shooting percentages,
efficiency ratings, pace) are left as-is.

**Leakage control.** Any column that directly contributes to the target is
removed, along with redundant columns: wins/losses and their ranks, `W_PCT`-derived
fields, `PLUS_MINUS`, and `NET_RATING`. These would let the model reconstruct the
target instead of learning from basketball stats. `FT_PCT` and similar are removed
because they're recoverable from makes and attempts, which are kept.

**One feature set, all 27 columns.** The real-team model and the fictional
path score on the same features. A smaller, rates-and-defence-only set was
built and tested for the roster path — on the argument that collinear
coefficients which only cancel in combination are unsafe on an extrapolated row
— and **rejected**: it fixed the coefficient signs and made aggregated-roster
predictions measurably worse (7.7 → 12.9 wins MAE), because pruning the
well-aggregated counting stats concentrates weight onto the two rebound-share
approximations this pipeline estimates worst.

**Coefficient signs are printed, never asserted.** 7 to 8 of the 16 features
with a defensible expected sign fit backwards — `OFF_RATING` at −0.88, `AST` at
−0.34, `TOV` at +0.34 — which costs nothing on real teams and is a live caveat
on a roster. `aggregate_team.coefficient_signs()` prints every coefficient with
its expectation annotated on each run. Nothing is asserted: at these magnitudes
several signs flip between fits (`PF` −0.003 → +0.014, `BLK` across zero), so an
assertion would pass or fail the identical model depending on which rows it saw,
and asserting a sign that can't be argued from first principles turns the test
suite into a device for tuning the model toward a prior.

**Every prediction carries a 90% interval.** Built on the logit scale and
transformed back, so it comes out correctly asymmetric near the bounds, and
reported as two separate components (model error, outcome randomness) plus
their combination — with the terms it does not cover named explicitly. See the
Test log.

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

> **These two numbers are stale and are not the logit change's doing.**
> `DEF_RATING` is pulled into `team_season_stats.csv` but excluded by
> `train_model.feature_columns()`, so the real-team model no longer sees it and
> is down to **MAE ≈ 0.059 → 4.85 wins, test R² 0.813** (the linear-target model
> scores 4.92 wins / R² 0.806 on the same rows — see the logit Test log entry,
> which reports both). That is the predicted consequence of removing one of the
> model's two dominant drivers, and the gap is the measurement of what it was
> worth: refitting the same 27 features **with** `DEF_RATING` gives 2.53 wins /
> R² 0.946, so the column is worth **2.32 wins** of MAE here. That figure is
> what the prediction interval prints as the error it does not cover. Re-measure
> and rewrite this section once the DEF_RATING decision is settled.

**Coefficient units changed with the target.** The standardized coefficients
`train_model.py` prints are now **log-odds per 1 SD**, not `W_PCT` per 1 SD.
There is no single win-equivalent for one of them, because the logistic's slope
depends on where the prediction sits; the printed `wins_at_500` column converts
at the steepest point (`dW_PCT/dz = p(1-p) = 0.25` at `p = 0.5`), which is the
largest swing a coefficient can produce, not a typical one.

## Test log

Each entry: *what was tested → what the evidence showed → what was decided.*
Decisions came from experiments, not intuition, and the ones that were reversed
are marked rather than deleted.

### `*_PCT` clipped to the contract, with the clip reported

*Tested.* The contract requires every `*_PCT` output to be a fraction in [0, 1],
and `SHARE_COLS` had no upper bound — the five-centre roster produced
`DREB_PCT` = **1.482**, a team securing 148% of the opponent's misses.

*Approach.* Two mechanisms, because the two kinds fail differently. The
opponent-dependent approximations (`OREB_PCT`, `DREB_PCT`) are **clipped**, with
every clip recorded in `result.attrs["pct_clipped"]` as `{col: pre-clip value}` so
a caller can tell a clipped line from a clean one. Every other `*_PCT` is
recomputed from the roster's own totals and cannot honestly leave range, so those
**raise** (`_check_pct_range`) — clipping them would turn a formula bug into a
plausible number.

*Evidence.* The clip fires on one of five stress rosters (five centres, 1.482 →
1.000) and no real team's aggregated rotation trips it. *Decision:* **integrated.**

### Stress rosters, with the expected ordering stated first

Five roster shapes built from prime seasons, each reporting `usage_scale`, the
predicted record and interval, the extrapolation ratio, and the three largest
contributions to the predicted log-odds. `EXPECTED_STRESS_ORDER` is written into
the source **before** the run, so a disagreement is a finding.

| Roster | `usage_scale` | Predicted | Ratio | Largest contribution |
|---|---:|---:|---:|---|
| balanced five | 0.696 | 80-2 | 3.0x | `DREB` +2.45 |
| 5 centres, no ball handling | **1.187** | 82-0 | **9.2x** | `REB` +6.92 |
| 5 low-usage role players | **1.467** | 68-14 | 3.7x | `PACE` +1.38 |
| 5 ball-dominant guards | 0.550 | 52-30 | 2.2x | `REB` −0.99 |
| 5 short off-ball shooters | 0.733 | 11-71 | 4.0x | `DREB` −2.42 |

*Findings, none tuned toward.* **(a)** Expected balanced ≳ centres > guards >
shooters; realised **centres > balanced** — the top two swap. **(b)** The
`usage_scale > 1` regime had never been exercised; the low-usage five was built to
hit it (1.467) and does, but so do the five centres (1.187), which was not
expected — five non-creators' summed volume falls short of their own pace anchor
just as five role players' does. **(c)** 82-0 for five centres is implausible and
traceable: their `DREB_PCT` reached the model only after clipping from 1.482, and
a clipped 1.000 reads as the best defensive rebounding any team has posted.
Nothing in the feature set encodes position, so the model can't know that roster
has no ball handler. *Decision:* recorded in Limitations.

### A 90% interval on every win-rate prediction

Three requirements, each of which the obvious implementation gets wrong.

**(1) Built on the logit scale** and transformed back — `inv_logit(z ± 1.645·sd)`
— so it comes out correctly asymmetric near the bounds. A symmetric band around a
predicted 0.95 runs past 1.0 and quotes wins that don't exist.

**(2) Two components, never blended into one number.** *Model error*: SD of
held-out residuals on the logit scale, **0.270** log-odds. *Outcome randomness*:
irreducible and exact — a season is 82 Bernoulli trials, so a realised rate has SD
`sqrt(p(1−p)/82)`. Combining needs a shared scale, so the outcome SD is carried to
log-odds by the delta method and the two add in quadrature. Reproduced exactly by
`prediction_interval()`:

| Predicted | Model error | Outcome only | Combined |
|---|---|---|---|
| 41 W | 32–50 | 34–48 | 30–52 |
| 57 W | 49–64 | 51–64 | 46–66 |
| 70 W | 64–74 | 64–75 | 61–75 |
| 78 W | 76–79 | 75–81 | 72–80 |

**(3) What it does not cover is printed with it, unquantified:** the dropped
`DEF_RATING` (worth **2.3 wins** of MAE on real teams — 4.85 without vs 2.53 with)
and the roster's extrapolation ratio. Neither is quantifiable from available data,
and inventing a term would be worse than having no interval.

*A degenerate case, flagged rather than printed.* As a prediction approaches a
bound, log-odds stretches without limit, so the combined band gets *wider* even as
the win counts pin — at full saturation it covers 0-82 while the model-error band
reads 82-82. That's the transform being honest, but printed bare it looks
measured, so `prediction_interval()` flags a band spanning ≥90% of the season and
`interval_report()` says so in words. The five-centre roster is the live case.
*Decision:* **integrated** everywhere a record is printed.

### Logit target, plus an extrapolation guard

*Tested.* `W_PCT` is a proportion bounded in [0, 1] and a ridge line has no stop
at either end. On real teams that never bites, but the five-star stress roster
predicted **1.004** with conservation on and **−2.609** with it off, and one *real*
aggregated team line came back at **−0.048**.

*Approach.* Fit `logit(p)` and invert with the logistic, via
`TransformedTargetRegressor`, so callers still pass and read plain `W_PCT` and the
bound becomes structural. `logit()` clips 1e-6 off each pole; the clip never binds
on real data (the 360-team span is 0.122–0.890).

*Evidence — accuracy is a wash, boundedness is not.*

| Rows scored | linear | logit |
|---|---:|---:|
| Chronological holdout | 4.92 W | **4.85 W** |
| Aggregated rosters, conservation ON | **7.4 W** | 7.5 W |
| Aggregated rosters, conservation OFF | 13.1 W | **12.5 W** |
| Five-star roster, ON / OFF | 1.004 / −2.609 | **0.881 / 0.000** |

Expected: the logistic is near-linear across the band real teams occupy, so on
real teams the transform has almost nothing to do. It earns its place at the
extremes, which is the whole point of the program. *Decision:* **integrated.**

*The cost of the fix, and the guard that pays it.* Bounding the output removes the
tell — a win rate of 2.44 announces itself as nonsense, a clean-looking 82-0 does
not. So `fit_extrapolation_guard()` measures Mahalanobis distance from the
training centre on standardized features (Ledoit-Wolf shrunk purely for
conditioning, since `PACE`/`POSS` correlate at ~0.99), with both thresholds read
off the real teams rather than a chi-square table: `edge` = 5.78 (99th pct),
`limit` = 6.36 (max). A per-feature bounding box would not do — an ordinary
five-man rotation breaks exactly one feature's range while landing 2.1x out,
because what's wrong with it is the *combination*.

*Calibrating the threshold.* Walking the guard forward one season at a time (8
folds, 240 unseen real teams), 4.2% of genuinely real teams stepped past the prior
seasons' maximum, and the worst reached **1.13x**. So a ratio near 1.0 means "a
team shape the league hadn't produced yet," and the flag's *magnitude* is the
signal.

*Evidence — the flag fires on everything, which is itself the finding.*

| Row shape | ratio to `limit` | flagged |
|---|---:|---:|
| Real team-season the guard never saw | ≤ 1.13x | 4.2% |
| Real team's own top-15 rotation, aggregated | 1.7x (max 2.4x) | **100%** |
| Real team's own top-5 rotation, aggregated | 1.9x (max 3.3x) | **100%** |
| Five-star roster, conservation ON | 5.1x | 100% |
| Cross-era five-star roster | 6.4x | 100% |
| Five-star roster, conservation OFF | 9.2x | 100% |

Every roster that goes through `aggregate_team()` lands outside the real-team
cloud — including a real team's own starting five, whose season actually happened.
The aggregation displaces a line by itself, before any fictional-ness enters,
mostly through the rescale to 240 minutes. *Decision:* keep the threshold defined
by the 360 real teams, but report the **ratio** as the headline with the measured
scale alongside, and record that the boolean is a constant on the roster path. A
flag that always fires is worth nothing; a ratio separating an ordinary rotation
(1.9x) from a stacked five (5.1x) is worth something.

### Ridge vs. gradient boosting (real-team model)

| Training rows | GB train R² | GB test R² | gap |
|---|---:|---:|---:|
| 300 (10 seasons) | 0.986 | 0.916 | 0.070 |
| 360 (12 seasons) | 0.981 | 0.919 | 0.062 |
| 540 (18 seasons) | 0.973 | 0.931 | 0.042 |

*Decision:* ridge ships. ~30 team-seasons per year is inherently small data, which
favours a simple regularized linear model. (GB lost again on the lineup defensive
model, with 10x the rows — see below.)

### `TM_TOV_PCT` formula bug

The aggregated turnover rate used a plays-based denominator (`FGA + 0.44·FTA +
TOV`) while the team CSV stores a possession-based one (`TOV / POSS`), biasing it
low on every team. Because it's a high-coefficient feature, this inflated
predicted wins across the board. *Decision:* fixed; aggregated win-MAE dropped
**14.3 → 6.5**. *Standing rule:* a derived rate must match the CSV column's exact
**definition**, not merely its units.

### `DEF_RATING` synthesis — four attempts, none shipped

| Attempt | Approach | Evidence | Decision |
|---|---|---|---|
| Sub-model | Predict team `DEF_RATING` from aggregatable defensive box-score stats | test R² ≈ 0.19; no better than dropping the column | not used |
| Personal composite | One feature regressing personal defensive stats on real `DEF_RATING` | 7.57 vs 7.60 wins MAE, and the edge traced to collapsing four collinear columns (a regularization effect), not recovered signal; vanished through aggregation | not used |
| Awards composite | Rule-based score from DPOY / All-Defensive selections | equal to or worse than dropping | not used |
| Prior-season lineup model | See the dedicated entry below | real but small (~4% of team-level MAE) | not integrated |

A fifth option scored **best** of everything tried (~6.5 wins) and was rejected:
a minutes-weighted average of the players' own real `DEF_RATING`. A player's
`DEF_RATING` is the *team's* points allowed while he was on the floor, so
averaging a team's players returns approximately that team's own rating — it
scored well *because* it was circular, and the circularity vanishes on a
cross-era roster.

*Final:* `DEF_RATING` is **dropped** from the fictional path. All routes failed for
one structural reason — it is opponent-dependent, and defense is largely composed
of events that *don't* happen (a shot not taken, a drive abandoned) while a box
score records events that do. The resulting ~7.6-win aggregated MAE is accepted as
an honest floor. (`train_model.py` also excludes it, so both paths share one
feature set.)

### Possession conservation

*Problem.* Summing five players' box-score volume can imply more possessions than
a game contains — the five-star roster implies ~181 against a real ~99.
*Approach.* Anchor the roster to a possession target = minutes-weighted average of
the players' own `PACE` (validated: tracks real team `PACE` at ~1.4% mean abs
error, corr 0.985), then scale volume to match, deriving `POSS` from the scaled
totals rather than setting it independently. *Evidence:* possession-denominated
rates (`OFF_RATING`, `TM_TOV_PCT`) are provably invariant to the scaling (ON vs OFF
differ by ~1e-14), so the mechanism adjusts volume without distorting efficiency;
`usage_scale` ≈ 1.0 on real teams (mean 1.0006) and **0.55** on the five-star
roster. *Decision:* **integrated.** A near-no-op on real teams by design; it bites
only where it should.

*Known asymmetry.* `usage_scale` applies to `USAGE_SCALE_COLS` only, so `DREB`,
`REB`, `STL`, `BLK` and `PF` take the 240-minute rescale without the pullback.
`PFD` was moved into the scaled family after `FTA`/`PFD` came out at 0.710 against
a real-team range of 0.953–1.303 — a foul drawn is what creates a free throw. The
rest were left alone as genuinely opponent-driven.

### Usage→efficiency response

*Tested.* Whether TS% can be adjusted for a usage change, using situation-change
season pairs (traded, or a high-usage teammate arrived/left) to isolate
externally-caused shifts, split by an on-ball/off-ball tracking index.
*Evidence:* real but weak, and **only for off-ball players** — off-ball slope
significant (usage down → efficiency up, p ≈ 0.005) but low-signal (R² ≈ 0.08);
on-ball slope indistinguishable from zero (p ≈ 0.61). Per-player predictions are
near coin-flips. *Decision:* **not integrated.** Any five-star roster forces usage
swings far beyond the observed range, where predictions run entirely on the
extrapolation taper. Kept as a documented experiment.

*Why the effect is small for everyone, which strengthens the decision:* the
rosters with the largest usage cuts are ball-dominant (five guards at
`usage_scale` 0.550) and sit in the **on-ball** bucket where the slope is null;
the rosters in the off-ball bucket where the slope is real have little usage to
lose (five shooters at 0.853). Big drops where there's no slope, real slope where
there's no drop.

### Perturbation test replacing the MAPE × coefficient ranking

*Problem.* The old `error_impact()` crossed per-feature aggregation MAPE with
standardized ridge coefficients. Unreliable under collinearity: `PACE` and `POSS`
correlate at ~0.99 with large opposite-signed coefficients, so a coefficient's
magnitude overstates how much *that feature's own* error moves a prediction.
*Approach.* `perturbation_impact()` perturbs each feature by its measured MAPE
(both directions), re-predicts, and averages the win-count change over all 30
teams — model-agnostic, so it survives the logit target and would survive a
nonlinear model. *Evidence:* the rankings disagree exactly where predicted, `PACE`
and `DREB_PCT` swapping 7–9 places. `OREB_PCT` has the worst MAPE (25%) but ranks
13th, while `PACE` at 1.35% ranks first — because `PACE`'s spread across real teams
is tiny, so a small percentage error is a large z-score. *Decision:*
`perturbation_impact()` is primary; the old function is kept as
`error_impact_deprecated()` so both print side by side. *Known limitation:* it
perturbs one feature at a time, assuming independent errors — false here, since
`PACE`, `POSS`, `FGA` and `PTS` errors all originate in the same usage-scaling
step.

### 5-man lineup validation (vs. the 15-man team validation)

*Why.* The 15-man validation rebuilds a real team from its top-15-by-minutes
roster, but the program is asked to score 5–15 players, most often 5.
*Approach.* `pull_lineups.py` pulls every real 5-man lineup's own observed line
(23,470 lineup-seasons, 2014-15 onward); each lineup clearing a 250-possession
floor has its five players' season stats run through `aggregate_team()` and
compared to the lineup's real line, possession-weighted (827 of 1,005 qualifying
lineups used; 178 skipped for the traded-player caveat).

*First pass:* rate features validated fine but every additive box-score total plus
`POSS` ballooned to 300–460% weighted MAPE. *Root cause:* `aggregate_team()`
rescales to a full 48-minute game while a real lineup's line covers only the
~15–20 minutes those five actually shared the floor — two incommensurable bases.
The 15-man validation never surfaced it because a real team's top-15 already sums
to ~240 combined minutes, making the rescale nearly a no-op; it only bites at the
roster size the program actually predicts. *Fix:* rescale the real lineup's line
up to a 48-minute-equivalent basis (`48 / MIN`), leaving rate columns alone.
*Evidence:* overall weighted MAPE **276.70% → 10.33%**, and `POSS` now tracks
`PACE` closely (1.81% vs 1.72%). `FTM`, `FTA`, `BLKA`, `PFD` and `BLK` still
validate worse at 5-man than 15-man (~18–24% vs ~3–6%) — the low-frequency events
where a 5-player sample is noisiest, a real remaining signal rather than a units
artifact. *Decision:* **rescale integrated.**

### 5-man lineup `DEF_RATING` from players' prior-season stats

The fourth and most careful `DEF_RATING` attempt, under a construction that makes
the earlier attempts' circularity *causally impossible*.

*Approach.* Predict a real lineup's observed `DEF_RATING` from its five players'
**prior-season** individual stats — a 2023-24 lineup described only by 2022-23
lines — with **all on-court team context forbidden from every season** (player
DEF/OFF/NET_RATING, `PLUS_MINUS`, on/off splits; `PIE` and `USG_PCT` also excluded
as not cleanly "his own"). 32 features, 27 after pruning, possession-weighted
throughout. Two rules, because they close different channels: the lag closes the
mechanical channel (a feature containing the label), the context ban closes the
confounding channel (a feature proxying for the label through a persistent team
scheme).

*Feature construction.* Five players are an unordered set, so features are pooled,
not concatenated by slot. At a fixed group size of five a **sum is exactly 5x the
mean**, so including both would make the design matrix exactly singular (mean only
is kept); and pooling is order-free in exact arithmetic but *not bit-identical* in
floating point, so players are sorted by `PLAYER_ID` first.
`test_permutation_invariance()` shuffles all five slots and asserts the feature
matrix is **byte-identical** — it passes (999 of 1,005 rows actually reordered).
Order statistics (max/min/std of `BLK_36`, `STL_36`, `DREB_PCT`, height) sit on top
of the means, since a mean hides "this lineup has one elite rim protector."

**Sample-weight scale silently disabled the ridge penalty.** *Found:* RidgeCV was
selecting α = 1000 — exactly the top of `train_model.py`'s grid — and Ridge and OLS
agreed to three decimals. One bug caused both. sklearn's objective is
`Σ wᵢ(yᵢ − xᵢ·β)² + α‖β‖²`, so weights and penalty share a scale; weights were raw
season-total possessions (mean ≈ 550), multiplying the data term by ~550 and
dividing the effective penalty by the same. α = 1000 was acting like α ≈ 2 — an
essentially unregularized fit, which is why "Ridge ≈ OLS" looked like evidence of
a robust signal and was nothing of the kind. *Fix:* mean-normalize the weights
(relative weighting, and every weighted metric, unchanged) and extend the grid to
`logspace(-3, 6, 40)`. *After:* α lands at 41–346, strictly **interior** every
time, and ridge actually shrinks (`‖ridge‖/‖OLS‖` 0.78 → 0.16 as the floor rises).
It also reversed the ranking — Ridge now beats OLS at every floor.
***Standing rule:* a CV-selected hyperparameter pinned to a grid endpoint is a bug
report, not a result.**

**Rolling-origin CV replaced the single 2-season holdout.** The old evaluation
trained on 10 of 12 seasons and reported one number from ~160 lineups — one
season's weather, not a measurement, and three conclusions drawn from it did not
survive. *Approach:* expanding chronological window, never shuffled — fold 1 trains
2014-15..2019-20 and tests 2020-21, out to 2025-26. 95% CIs are computed **across
folds** (t, df = 5), not by bootstrapping rows: lineups within a team-season share
four of five players, so a row bootstrap would be anti-conservative.

*Headline metric is the season-centered target.* League-average team `DEF_RATING`
rose **104.7 → 114.7** across the sample, so a raw-target R² is largely charged for
a league-wide scoring shift unrelated to which five players are on the floor.
Centering isolates the lineup question; using it in production would require
forecasting next season's league average separately.

*Possession floor, re-picked under the CV folds* (Ridge, season-centered):

| POSS floor | lineups (OOF) | R² ceiling | R² mean [95% CI] | team slope | team-seasons | team MAE (cal.) vs league avg |
|---:|---:|---:|:--|---:|---:|:--|
| 50 | 6,446 (3,177) | 0.167 | **+0.026 [+0.016, +0.037]** | 0.716 | 179 | 2.24 vs 2.23 (worse) |
| 100 | 2,621 (1,227) | 0.227 | +0.035 [−0.003, +0.073] | 0.712 | 166 | 2.17 vs 2.19 |
| 200 | 1,047 (477) | 0.285 | +0.058 [−0.005, +0.120] | 0.708 | 142 | 2.05 vs 2.11 |
| 250 | 757 (335) | 0.330 | +0.058 [−0.006, +0.122] | 0.705 | 133 | **1.96 vs 2.07** |

*Evidence:* more data is not simply better — at floor 50 roughly 83% of the
label's variance is sampling noise and the team-level estimate is **no better than
predicting the league average**. *Decision:* `PRIMARY_POSS_FLOOR = 250`, chosen on
the team-level column; floor 200 is statistically indistinguishable with 42% more
held-out lineups and is the conservative alternative. Two caveats: above floor 50
the lineup-level CI **includes zero**, and a high floor leaves 133 of 180
team-seasons, so the calibration is fit on a self-selected set of stable rotations.

*Model comparison* (floor 250, six folds, season-centered, MAE in `DEF_RATING`
points):

| Model | R² mean [95% CI] | worst fold | MAE | pooled-OOF R² |
|---|:--|---:|---:|---:|
| **Ridge** (α 70–346, interior) | **+0.058 [−0.006, +0.122]** | −0.027 | **4.52** | **+0.096** |
| OLS | +0.003 [−0.111, +0.117] | −0.131 | 4.66 | +0.048 |
| Gradient boosting (defaults) | −0.066 [−0.152, +0.020] | −0.165 | 4.82 | −0.026 |
| Baseline: constant (league avg) | −0.009 | −0.036 | 4.73 | +0.025 |
| Baseline: BLK+STL+DREB sum | −0.005 | −0.034 | 4.75 | +0.028 |

The naive blocks+steals+rebounds sum is worth **nothing** over a constant — a
useful negative result about the raw defensive box score. GB is negative on 4 of 6
folds despite ~10x the team model's rows and a domain where threshold effects
should live. The lesson from both GB losses: what matters is *signal* per
parameter, not rows per parameter, and 67–83% of this label is noise.

*Roster continuity: a null, and a retraction.* The single-holdout run had shown
continuity lineups defending ~1.5 points better than their players' prior stats
predict and reshuffled ones ~1.1 worse — a 2.7-point bias spread that looked like a
real chemistry effect. `CONTINUITY_PAIR_FRAC` (of the 10 pairs among five players,
the fraction who ended the prior season on the same team — membership only, so it
is computable for a roster that never played) is a **null**: coefficient −0.073
points per SD, R² 0.058 with it vs 0.057 without. And the effect it was built for
largely **was not there** — under CV the bias spread is 0.99 points, not 2.7, and
the MAE gap *reverses sign* (mover lineups predicted **better**, 4.35 vs 4.72). A
per-group intercept moves the gap by 0.001, so there is no sign of context
memorization. *Decision:* kept (it costs nothing and is the right shape for the
question) but recorded as a null; the 2.7-point claim is **withdrawn as
single-holdout noise**.

*Sensitivity, then pruning.* Top of the ±1-SD ranking: `BLKA_36_mean` (0.44
pts/SD), `PCT_PLUSMINUS_mean` (0.35), `BLK_36_min` (0.32), `BLK_36_max` (0.27),
`BLK_36_mean` (0.27), `STL_36_mean` (0.25) — shot-blocking and closest-defender
columns, the right shape for a defensive model. Pruning the three collinear blocks
the coefficient table exposed took 32 features to 27 with R² 0.058 → 0.051. *A
wash to slightly negative*, every step far inside the ±0.06 CI — ridge was already
handling the collinearity, which is what L2 is for. *Decision:* keep the 27-feature
set on parsimony grounds, with the ~0.007 cost recorded rather than presented as an
improvement.

*Reliability / calibration (required before integration).* Out-of-fold predictions
from all six held-out seasons, rolled up to team level, regressing true team
`DEF_RATING` on the estimate, 133 team-seasons:

| Model / target | slope | R² | team MAE raw → cal. | league-avg baseline |
|---|---:|---:|:--|---:|
| **Ridge, season-centered** | **0.679** | 0.233 | 2.93 → **1.98** | 2.07 |
| OLS, season-centered | 0.483 | 0.202 | 3.10 → 2.01 | 2.07 |
| Ridge, raw target | 0.493 | 0.173 | 5.28 → 2.06 | 2.07 |

`DRtg_used = 37.85 + 0.679 × DRtg_est` is what a future integration must apply,
and the slope is stable across floors (0.705–0.716), which is what a real
coefficient looks like. *Read the R² with care — it uses the wrong baseline.*
`r2_score` compares against the **pooled** mean, but those team-seasons span six
seasons of ~5 points of league drift and the centered model adds each season's
average back into every prediction, so part of 0.233 is credit for tracking drift.
Against a per-season baseline the MAE ratio implies **R² ≈ 0.085**. The MAE
comparison is the number to quote.

*Decision:* **not integrated.** 1.98 vs 2.07 is a 4% improvement — about **0.2
wins** at ~2.5 wins per point of net rating — which cannot carry the weight of
replacing a column the win model treats as one of its two dominant inputs.

*Hustle stats: null under CV.* Contested shots, deflections, box-outs, charges and
loose balls exist only from 2016-17 (2015-16 returns a 147-player, median-1-game
partial season that *looks* like real data), so using them costs three lineup
seasons. On the identical reduced sample and folds: R² **0.030 [−0.224, +0.285]
with** hustle vs **0.039 [−0.224, +0.303] without**. *Decision:* **off by default**,
with the comparison printed every run. (The single-holdout run had made hustle look
like a 0.23-point gain — another number that did not survive.)

*Rookies.* A lineup is dropped if any of its five players lacks a usable prior
season, rather than imputed — a league-average defender would inject a fabricated
feature vector under a real label. This is the largest sample cut: 230 of 1,005
lineups (23%) at floor 250, 34% at floor 50. See Limitations for the resulting
bias.

## Limitations & next steps

- **No ground truth for the end goal.** Fictional/cross-era teams never played,
  so their predictions are informed estimates that can't be directly verified.
  The pipeline is graded only *indirectly*, on real teams.
- **A bounded prediction is not a supported one.** The logit target guarantees
  every predicted record is *possible*; it guarantees nothing about whether it
  is *right*. A saturated 82-0 is the logistic running out of room, and it now
  looks exactly as reasonable as a 48-34. The extrapolation guard exists to
  restore the distinction the old impossible numbers used to make for free, and
  it should be read every time — never quote a fictional record without the
  ratio next to it.
- **The interval covers the two smallest terms, and says so.** Model error
  (0.270 log-odds) and 82-game outcome randomness are the two components that
  *can* be measured, so they are the two the interval contains. The dropped
  `DEF_RATING` is worth ~2.3 wins of MAE on real teams on its own, and the
  extrapolation — 2.2× to 9.2× for the stress rosters — has no mapping to wins
  at all. Both are printed next to every interval and neither is inside it. A
  90% interval on a fictional roster is therefore a *lower bound* on the
  uncertainty, not a calibrated one.
- **Several coefficients are backwards, and rosters are where that could
  bite.** 7 to 8 of the 16 features with a defensible expected sign fit with the
  wrong one — `OFF_RATING` −0.88, `AST` −0.34, `TOV` +0.34 — because collinear
  features carry offsetting values that cancel only while they keep moving
  together. Real teams always satisfy that; an aggregated roster is not
  guaranteed to, and usage conservation deliberately breaks one of the
  relationships involved (it scales `PTS` while `POSS` is derived from the
  scaled totals).
- **The stress rosters expose two things the model gets wrong.** Five centres
  predicts 82-0 at a 9.2× extrapolation ratio, on `REB` and `DREB` totals from
  five great rebounders rescaled to 240 minutes, with `DREB_PCT` reaching the
  model only after being clipped from 1.482. Five low-usage defensive
  specialists predict 61-21, which is generous for a roster with no scoring —
  its largest positive term is `PACE`, the model's biggest coefficient (−0.98
  per SD) and one with no defensible expected sign. The realised ordering also
  disagrees with the ordering written down before the run (centres and the
  balanced five swap at the top). None of this was tuned away.
- **Every aggregated roster is an extrapolation, so the guard's flag is a
  constant.** Effectively 100% of rosters put through `aggregate_team()` land
  beyond the furthest of 360 real team-seasons — including real teams' own
  top-5 (median 1.9×) and top-15 (median 1.7×) rotations, whose seasons
  actually happened. (The current season's 30 teams put top-5 at 29 of 30, so
  "always" is a strong tendency rather than a law.)
  The aggregation displaces a line off the real-team cloud on its own, mostly
  through the rescale to 240 minutes. Only the *ratio* discriminates
  (ordinary rotation ~1.9× vs. stacked five ~5.1×), and it is a rough,
  monotone "how strange is this" measure, not a calibrated error bar — nothing
  maps a ratio to an expected number of wins wrong. Building that mapping (or
  re-anchoring the guard on aggregated real rosters instead of real team lines,
  so the flag means "strange for a roster" rather than "strange for a team")
  is the obvious next step and is not built.
- **DEF_RATING is dropped on the fictional path.** The part of defense that lives
  in scheme and opponent behavior can't be manufactured from a roster's own
  stats by any method tested (see Test log). This is the largest single component
  of the ~7.6-win aggregated error floor. The prior-season lineup model
  (`lineup_defense_model.py`) is the closest thing to a non-circular estimate the
  project has, and it is still not good enough to wire in — see the four
  limitations immediately below.
- **Lineups are not randomly assigned — the target is selection-biased.** Coaches
  choose which five players play together, and when. Good lineups get more
  minutes, start halves, face opposing starters, and appear in different game
  states (score margin, clock, opponent personnel) than bench units do; a
  closing lineup's DEF_RATING partly reflects that it was on the floor in
  close, slow, half-court games. So a lineup's observed DEF_RATING is not a
  clean measurement of "how well these five defend" — it is that, plus the
  context its coach put it in. Possession-weighting and a large sample dilute
  this (each team's most-used units carry the most weight, and coaching
  tendencies differ across 30 teams and 12 seasons) but do **not** remove it:
  the bias is correlated with lineup quality, which is exactly the thing being
  predicted. Nothing in the current model corrects for it; an
  opponent-and-game-state adjustment would be the honest fix.
- **Every R² here has to be read against the label-noise ceiling, not against
  1.0.** A lineup's DEF_RATING is a sample mean over its possessions, so most of
  its observed spread is sampling noise in the *label*, not signal a model could
  ever recover. Decomposing that (`label_noise_decomposition()`: fit
  `Var_observed = Var_true + k·mean(1/POSS)` across possession bins,
  season-centered) puts the achievable ceiling at **0.167 at a 50-possession
  floor, 0.227 at 100, 0.285 at 200, 0.330 at 250** — i.e. even a model that knew
  each lineup's true defensive quality exactly would score R² ≈ 0.33 at the floor
  used here. The model's 0.058 is therefore ~18% of what is available, not ~6% of
  a perfect prediction. Quoting a lineup-level R² without the ceiling next to it
  makes an ordinary result look like a failure and a noise-floor artifact look
  like signal.
- **The season-centered result at the chosen floor is R² 0.058 [−0.006, +0.122],
  and the CI includes zero.** Six rolling-origin folds, Ridge, floor 250: positive
  on 5 of 6 held-out seasons (worst −0.027, best +0.141), MAE 4.52 vs the
  league-average baseline's 4.73. The interval is wide because the honest unit of
  replication is a *season*, not a lineup (see non-independence below), and six
  seasons is six data points. Only the 50-possession floor produces an interval
  that excludes zero (+0.016 to +0.037) — and that floor's team-level estimate is
  no better than predicting the league average. Read the lineup-level effect as
  "small, probably real, not yet separable from zero at this sample size."
- **The reliability slope is 0.68 — about a third of the estimate's spread has to
  be shrunk away.** Out-of-fold across all six held-out seasons, rolled up to
  team level (possession-weighted), true team DEF_RATING on the estimate:
  `DRtg_used = 37.85 + 0.679 × DRtg_est`, R² **0.233**, 133 team-seasons,
  sd(estimate) 2.00 vs sd(true) 2.82. Any future integration must apply that
  slope and intercept, not the raw estimate. In MAE terms the calibrated team
  estimate is **1.98 vs 2.07** for "just predict the league average" — a 4%
  improvement, because R² 0.233 only cuts residual SD by ~12%. That combination
  (a real, stable slope; a barely-better MAE) is the whole reason the model is
  documented and *not* wired in. The earlier 0.394 slope / R² 0.112 in this
  README came from the buggy-ridge single-holdout run and is superseded.
- **The team-change gap does not reproduce.** On a single 2-season holdout the
  model looked 0.79 MAE points worse on lineups containing a player who changed
  teams, with a 2.7-point opposite-signed bias split — which read as a real
  continuity effect. Under six rolling-origin folds (335 held-out lineups instead
  of 159) the sign **reverses**: mover lineups are predicted *better* (MAE 4.35 vs
  4.72, R² 0.122 vs 0.050) and the bias spread is 0.99 points, not 2.7. A
  per-group intercept moves the MAE gap by 0.001, so there is no evidence of the
  model memorizing team context — but also no measured continuity penalty to
  apply to a fictional roster. What remains is a genuine caution of a different
  kind: `CONTINUITY_PAIR_FRAC` = 0 (five players who never shared a team, i.e.
  every cross-era roster) describes **0.4%** of the training sample, so the model
  is extrapolating there regardless of how small its continuity coefficient is.
- **Rookie exclusion biases the sample toward veteran lineups.** A lineup is
  dropped whenever any of its five players has no usable prior season, which
  removes 23% of lineups at the primary possession floor (34% at floor 50).
  Dropping rather than imputing keeps fabricated feature vectors out of the fit,
  but the surviving sample systematically over-represents established rotations
  and under-represents young, high-turnover lineups — so the model has been
  fit and graded on veteran units, and its accuracy on a lineup built around
  first- or second-year players is unmeasured, not merely worse. The same cut
  removes team-seasons entirely at high floors: the calibration above is fit on
  133 of 180 possible team-seasons, self-selected toward stable rotations.
- **Lineup rows are far from independent.** Within one team-season, dozens of
  lineups share four of their five players, and a player's prior-season line is
  reused by every lineup he appears in. Row counts therefore overstate the
  effective sample size considerably. Three consequences, all live: CIs are
  computed across season folds rather than by bootstrapping rows (a row
  bootstrap would be anti-conservative); the evaluation is chronological and
  never random; and the gradient-boosted fit's train R² should be read as
  memorization rather than signal.
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