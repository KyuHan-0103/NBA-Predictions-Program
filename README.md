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
costing accuracy where it counts (Test log). Every prediction is reported with
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

| File | Purpose |
|------|---------|
| `pull_team_stats.py` | Pulls Base + Advanced team stats, merges on TEAM_ID + SEASON, converts counting stats to per-game, writes `team_season_stats.csv`. Data only. |
| `pull_player_stats.py` | Pulls current-season per-player Base + Advanced stats → `player_season_stats.csv`. Data only. |
| `pull_player_seasons_all.py` | Pulls per-player stats for every season 1996-97 onward (the roster-input pool) → `player_all_seasons.csv`. Data only. |
| `pull_lineups.py` | Pulls 5-man lineup stats (Base + Advanced), 2014-15 onward → `lineup_season_stats.csv`. Ground truth for `aggregate_team.py`'s 5-man validation. Data only. |
| `pull_player_defense.py` | Pulls per-player *individual* defensive descriptors — bio (height/weight/age), closest-defender tracking (opponent FG% on shots he defended, 2013-14 onward), hustle (2016-17 onward) → `player_defense_stats.csv`. Data only. |
| `train_model.py` | Loads the team CSV, splits by season, trains the ridge win-rate model on `logit(W_PCT)`, prints metrics and standardized coefficients (the model's top drivers). Also home to `logit`/`inv_logit`, `ridge_step()` (reaches the `RidgeCV` inside the target-transform wrapper), the extrapolation guard every roster prediction is reported against, and `prediction_interval()` / `interval_report()`. |
| `aggregate_team.py` | Aggregates 5–15 players into one team stat-line (possession conservation, DEF_RATING dropped, `*_PCT` clipped to the contract), and self-validates by rebuilding every real team from its own roster. Also home to `coefficient_signs()` (expected vs. fitted signs, printed) and the stress rosters. |
| `perturbation_tests.py` | The aggregation's per-feature error-impact diagnostics (`perturbation_impact()` + the deprecated MAPE × coefficient ranking), split out of `aggregate_team.py`. Runs standalone, and `aggregate_team.py` still prints the identical tests via `run_perturbation_tests()`. |
| `predict_fictional_roster.py` | Interactive roster builder (by player + season); aggregates and scores the roster with the ridge model. |
| `usage_efficiency_model.py` | **Standalone, not integrated.** Fits and tests the usage→efficiency (USG%→TS%) response model. Kept as a documented experiment. |
| `lineup_defense_model.py` | **Standalone, not integrated.** Estimates a 5-man lineup's DEF_RATING from its players' prior-season individual stats (permutation-invariant pooling, possession-weighted, rolling-origin CV by season), with the possession-floor sweep, label-noise ceiling, both baselines, and the reliability / team-change diagnostics. Kept as a documented experiment. |
| `SPEC.md` | Full project specification. |
| `CLAUDE.md` | Working agreements, incl. the aggregation contract. |

## Setup & commands

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python pull_team_stats.py            # team data      -> team_season_stats.csv
python pull_player_seasons_all.py    # player pool    -> player_all_seasons.csv
python pull_lineups.py               # 5 man lineups  -> lineup_season_stats.csv
python pull_player_defense.py        # player defense -> player_defense_stats.csv
python train_model.py                # train + report the real-team model
python aggregate_team.py             # aggregation + real-team validation
python perturbation_tests.py         # aggregation error-impact diagnostics only
python predict_fictional_roster.py   # build a roster and predict its record
python lineup_defense_model.py       # 5-man lineup DEF_RATING experiment
```

## Configuration

Knobs live at the top of each script:

- `NUM_SEASONS` — how many seasons of team data to pull (`pull_team_stats.py`)
- `N_TEST_SEASONS` — recent seasons held out for testing (`train_model.py`)
- `ALPHAS` — ridge penalty grid searched by cross-validation
- `MIN_GP`, `MIN_MPG` — player eligibility thresholds derived from graphs in docs (`aggregate_team.py`)
- `POSS_FLOORS`, `PRIMARY_POSS_FLOOR` — lineup possession floors compared, and the one chosen from that comparison (`lineup_defense_model.py`)
- `MIN_TRAIN_SEASONS`, `LINEUP_ALPHAS` — first rolling-origin fold's training depth, and the ridge grid (deliberately wider than `train_model.py`'s — see Test log) (`lineup_defense_model.py`)

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
approximations this pipeline estimates worst. See the Test log.

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

**A separate feature set for the fictional path — built, measured, and
reverted.** The 27-feature model's coefficients do not read as basketball:
`OFF_RATING` fits at **−0.877** log-odds per SD, `AST` at −0.536, `TOV` at
**+0.550**, and 7 of 16 priced features come out backwards. Three exact or
near-exact linear dependencies cause it — `PTS == 2·FGM + FG3M + FTM` (max abs
diff 4.3e-14), `POSS ≈ FGA + 0.44·FTA + TOV − OREB` (corr 0.988), `REB == OREB +
DREB` — plus `OFF_RATING ≡ 100·PTS/POSS` at corr 0.999981. Ridge cannot remove a
dependency; it picks one of infinitely many equally-good coefficient splits, and
that split is only valid where the dependency holds as it does in training.

*The hypothesis:* an aggregated roster is where those offsetting pairs stop
cancelling, because usage conservation scales `PTS` while `POSS` is derived from
the scaled totals — so a feature set without the redundancy should predict
rosters better even if it fits real teams slightly worse.

*What was built.* A 12-column rates-and-defence set (`OFF_RATING`, `EFG_PCT`,
`TM_TOV_PCT`, `AST_RATIO`, `OREB_PCT`, `DREB_PCT`, `PACE`, `STL`, `BLK`, `BLKA`,
`PF`, `PFD`) with four sign assertions raising `SignViolation` at fit time. Note
that pruning only `PTS`/`POSS`/`REB` does **not** work: those columns are
reconstructable from the ones that remain, and `OFF_RATING` still fits at −0.40
with 5 of 8 signs wrong. The counting stats and the derived rates span the same
space, so one side of the span has to go.

*It worked on its own terms.* 0 of 11 expected signs wrong instead of 8 of 16,
for 0.23 wins of holdout MAE (5.08 vs 4.85).

*Why it was reverted — the hypothesis was tested and failed.* Scored against
real teams' actual win totals from their own aggregated 15-man rosters — the only
end-to-end validation this project has — the smaller set is **twice as bad**:

| Feature set | Holdout MAE | R² | Aggregated rosters vs actual wins |
|---|---:|---:|---:|
| **27 (shipped)** | **4.85 W** | **0.813** | **7.6 W** |
| 12 (reverted) | 5.08 W | 0.783 | 15.2 W |
| variant: `DREB` for `DREB_PCT` | 3.97 W | 0.864 | 10.0 W |

The mechanism is visible in the aggregation's own per-feature validation: pruning
15 counting stats that aggregate at under 5% MAPE concentrates the model's weight
onto `DREB_PCT` (7.5% MAPE) and `OREB_PCT` (25.3%) — the two opponent-dependent
approximations, and the only two the aggregation itself flags low-confidence. The
stress rosters moved the same way: five All-Star guards 52-30 → 28-54, five
low-usage role players 68-14 → 31-51, five elite shooters 11-71 → **1-81**.

*And the one apparent win did not survive a control.* The 12-feature set appeared
to fix the guard's always-fires problem: median ratio 1.70x → 1.08x, flagged 100%
→ 65%. But scoring **random 12-column subsets** of the same 27 gives a median of
**0.92x** (range 0.60–1.25x over 8 draws) — *better* than the deliberately chosen
12. Mahalanobis distance in 12 dimensions is simply smaller than in 27: fewer
dimensions, fewer chances to be an outlier. The guard improvement was a
dimensionality artifact, not evidence about the feature set.

*Decision:* **reverted.** The fictional path is back on all 27 features. The
theory and the measurement pointed opposite ways and the measurement won — which
is the same rule that retracted the continuity effect, the hustle-stat gain and
the possession-floor choice earlier in this log. The one honest defence left for
the 12 is that aggregated real 15-man rosters sit at 1.7x the guard limit while a
cross-era five sits at 7–9x, so the test does not reach the regime the tool
actually operates in — but "better in a regime I cannot measure" is not a claim
this project accepts from itself.

*What was kept.* The `*_PCT` contract clip (Task 1), the stress rosters (Task 3),
the prediction intervals (Task 4), and `coefficient_signs()` — printed on every
run, nothing asserted. The sign problem is real and unresolved; printing it at the
point of use is the honest response to a fix that cost more than it bought. The
`DREB`-for-`DREB_PCT` variant is the one configuration that dominated the 12 on
every axis and is recorded below as the thing to test first if this is revisited.


The decisions below were made from experiments, not intuition. Each is recorded
as *what was tested → what the evidence showed → what was decided*, so the
reasoning survives and isn't rebuilt from scratch.

**A separate, smaller feature set for the fictional path — tested and
rejected.** The 27-feature model's coefficients do not read as basketball. Fit
on real teams, `OFF_RATING` comes out at **−0.88**, `DREB_PCT` at −0.12, `AST`
at −0.34, `TOV` at **+0.34** — 7 to 8 of the 16 features with a defensible
expected sign come out backwards, depending on which seasons the fit sees. That
costs nothing on real teams (test MAE 4.85 wins, R² 0.813): the offsetting pairs
cancel because the features move together in every row the model was fit on. A
roster is the row where they stop moving together, and there each coefficient is
applied on its own. So a feature set chosen to avoid that was built, measured,
and **not kept** — the reasoning and the numbers are below because the concern
is real even though the fix was worse.

*Why pruning is not the fix.* The obvious repair — drop `PTS`, `POSS`, `REB` —
does not work, because those columns are reconstructable from the ones that
remain: `PTS == 2*FGM + FG3M + FTM` (exact, max abs diff 4.3e-14), `POSS ~= FGA
+ 0.44*FTA + TOV − OREB` (corr 0.988), `REB == OREB + DREB` (exact). With all
three dropped, `OFF_RATING` still fits at −0.40 and `DREB_PCT` at −0.12: 5 of 8
signs still wrong. The counting stats and the derived rates span the same
space; keeping both leaves the rates free to take arbitrary offsetting values.

*What was tried.* Rates and defence only, no raw offensive counting stats:
`OFF_RATING, EFG_PCT, TM_TOV_PCT, AST_RATIO, OREB_PCT, DREB_PCT, PACE, STL,
BLK, BLKA, PF, PFD`. Measured on the same chronological holdout:

| Feature set | test MAE | test R² | expected signs wrong |
|---|---:|---:|---:|
| **full 27 (shipped)** | **4.85 W** | **0.813** | 8 of 16 |
| 9 (rates + STL/BLK) | 5.41 W | 0.747 | 1 of 8 (`AST_RATIO` −0.03) |
| 12 (rates + defence) | 5.08 W | 0.783 | 0 of 11 |

It worked as intended on its own terms: the 12-column set's coefficients read
as basketball (`EFG_PCT` +0.24, `STL` +0.17, `DREB_PCT` +0.16, `OFF_RATING`
+0.15, `BLKA` −0.13, `TM_TOV_PCT` −0.08), for 0.23 wins of holdout MAE.

*Why it was rejected: the 0.23 wins was not the price.* Scored against real
teams' actual win totals, from their own aggregated 15-man rosters — which is
the closest thing this project has to a test of the roster path — the same
aggregated lines give **7.7 wins MAE under the 27-feature model and 12.9 under
the 12**. The mechanism is visible in the aggregation's own per-feature
validation: `OREB_PCT` (25.25% MAPE) and `DREB_PCT` (7.47%) are the two
worst-aggregated features in the entire set — the opponent-dependent
rebound-share approximations, the only two the aggregation itself flags
low-confidence — and pruning 15 well-aggregated counting stats (most under 5%
MAPE) concentrates the model's weight onto exactly them. `DREB_PCT` became the
largest single log-odds contributor for four of the five stress rosters.
*Decision:* **not used.** The fictional path scores on all 27 features, the
same set `train_model.py` fits.

*What the rejection does not resolve.* The 7.7 is produced by coefficients that
are individually wrong and only correct in combination, on rows where that
combination is not guaranteed to hold — which is exactly the objection the
smaller set was built to answer, and it is unanswered rather than refuted.
What the experiment established is that the obvious fix trades one known
problem for a larger measured one. `aggregate_team.coefficient_signs()` now
prints every coefficient with its expected sign annotated on each run, so the
caveat is visible at the point of use; nothing is asserted, since a sign that
cannot be argued from first principles is not a thing to fail a build over.
A better answer would improve the rebound-share approximation first, which
would raise the ceiling on both feature sets at once.

*A footnote on why nothing here is asserted.* `PF` fits at −0.003 on the
chronological-holdout training rows and **+0.014** refit on all 12 seasons; the
27-feature model's wrong-sign count is 8 on one fit and 7 on the other, because
`BLK` crosses zero too. At those magnitudes the sign is noise, and an assertion
on it would pass or fail the identical model depending on which rows it saw.

**`*_PCT` clipped to the contract, with the clip reported.** The aggregation
contract requires every `*_PCT` output to be a fraction in [0, 1], and
`SHARE_COLS` had no upper bound: the five-centre stress roster produces
`DREB_PCT = 1.482`, a team securing 148% of the opponent's misses. *Approach:*
two mechanisms, because the two kinds of `*_PCT` column fail differently. The
opponent-dependent approximations (`OREB_PCT`, `DREB_PCT`) are **clipped**, and
every clip that fires is recorded in `result.attrs["pct_clipped"]` as
`{col: pre-clip value}` so a caller can tell a clipped line from a clean one.
Every other `*_PCT` is recomputed from the roster's own totals and cannot
honestly leave the range, so those **raise** (`_check_pct_range`) — clipping
them would convert a formula or units bug into a plausible-looking number and
leave the extrapolation report to notice, which is exactly the failure the
contract exists to prevent. *Evidence:* of the five stress rosters, the clip
fires on one (five centres, `DREB_PCT` 1.482 → 1.000) and none of the others;
no real team's aggregated rotation trips it. *Decision:* **integrated.**

**Stress rosters, with the expected ordering stated first.** Four lopsided
roster shapes plus a balanced five, all built from prime seasons
(`STRESS_ROSTERS`), each reporting `usage_scale`, the predicted record and
interval, the extrapolation ratio, and the three largest per-feature
contributions to the predicted log-odds. `EXPECTED_STRESS_ORDER` is written
into the source **before** the run, so a disagreement is a finding and not an
observation. Expected: balanced ≳ centres > ball-dominant guards > shooters.

| Roster | `usage_scale` | predicted | ratio | largest contribution |
|---|---:|---:|---:|---|
| balanced five | 0.696 | 80-2 | 3.0x | `DREB` +2.45 |
| 5 centres, no ball handling | **1.187** | 82-0 | **9.2x** | `REB` +6.92 |
| 5 low-usage role players | **1.467** | 61-21 | 3.7x | `PACE` +1.38 |
| 5 ball-dominant guards | 0.550 | 47-35 | 2.2x | `REB` −0.99 |
| 5 short off-ball shooters | 0.733 | 10-72 | 4.0x | `DREB` −2.42 |

*Findings, none of them tuned toward.* **(a)** The realised ordering is
centres > balanced > guards > shooters: the top two swap. **(b)** The
low-usage five was included to exercise the one regime nothing else had —
`usage_scale` **above** 1.0, the aggregation scaling volume *up* rather than
down — and it does (1.467), but so do the five centres (1.187), which was not
expected: five non-creators' summed volume falls short of their own pace anchor
just as five role players' does. The scale-up path had never been exercised
before this and now has two cases. **(c)** The five-centre prediction (82-0, at
a 9.2x extrapolation ratio) is implausible as basketball and is documented as a
limitation, not a target. Its two largest contributions are `REB` and `DREB` —
rebounding volume, rescaled to 240 minutes for five of the best rebounders who
ever played — and its `DREB_PCT` reached the model only after being clipped from
1.482 to 1.000. **(d)** Five low-usage defensive specialists predict 61-21,
which is generous for a roster that cannot score; `PACE` is its largest positive
term, and `PACE` carries this model's single largest coefficient (−0.98 per SD)
with no defensible expected sign. *Decision:* recorded in Limitations.

**A 90% interval on every win-rate prediction.** A predicted record with no
interval invites being read as precise; an interval built the obvious way is
worse, because it looks quantified while omitting its largest term. *Approach:*
three requirements, each of which the obvious implementation gets wrong.
**(1)** Built on the **logit scale** and transformed back —
`inv_logit(z ± 1.645·sd)` — so it is correctly asymmetric near the bounds. A
symmetric band in `W_PCT` around a predicted 0.95 runs past 1.0 and quotes wins
that do not exist. **(2)** Two components reported **separately**, never
blended into one number: *model error*, the SD of held-out residuals on the
logit scale (**0.270** log-odds), and *outcome randomness*, which is
irreducible and exact — a
season is 82 Bernoulli trials, so a realised rate has SD `sqrt(p(1-p)/82)`.
Combining them requires a shared scale, so the outcome SD is carried to
log-odds by the delta method (`1/sqrt(82·p·(1−p))`) and the two add in
quadrature. Reference values, reproduced exactly by `prediction_interval()`:

| Predicted | Model error | Outcome only | Combined |
|---|---|---|---|
| 41W | 32–50 | 34–48 | 30–52 |
| 57W | 49–64 | 51–64 | 46–66 |
| 70W | 64–74 | 64–75 | 61–75 |
| 78W | 76–79 | 75–81 | 72–80 |

**(3)** What it does *not* cover is printed with it and deliberately left
unquantified: the dropped `DEF_RATING` (worth **2.3 wins** of MAE on real
teams — 4.85 without vs 2.53 with) and the roster's own extrapolation ratio.
Neither is quantifiable from available data, and inventing a term for them
would be the one failure mode worse than having no interval at all.

*A degenerate case the delta method has, and how it is reported.* As a
prediction approaches a bound, log-odds stretches without limit, so the
combined band widens rather than narrowing — at full saturation it covers 0-82
while the model-error band still reads 82-82. That is the transform being
honest (a saturated prediction has no resolution left), but printed bare it
looks like a measurement. `prediction_interval()` flags the symptom
(`degenerate`: the combined band spans ≥90% of the season) and `interval_report()`
says so in words instead of quoting the range. The five-centre roster is the
live case. *Decision:* **integrated**, on both `predict_fictional_roster.py`
and every stress roster.

**Logit target for the win-rate model, plus an extrapolation guard.** `W_PCT` is
a proportion bounded in `[0, 1]`, and a ridge line fit directly on it has no
stop at either end. On real teams that never bites — every team-season sits in
`0.122–0.890`, and predictions stay inside the interval — but the fictional path
is exactly where it does. *Approach:* fit `logit(p) = ln(p / (1-p))` and invert
with `1 / (1 + e^-z)`, via `TransformedTargetRegressor`, so `build_model()`'s
signature is unchanged (callers still pass and read plain `W_PCT`) and the bound
becomes structural rather than hoped for. `logit()` clips at `1e-6` off each
pole; the clip never binds on real data (no 0-82 or 82-0 season exists in the
sample) and is there so the transform is total.

*Evidence — accuracy is a wash, boundedness is not.* Both models fit on
identical rows, scored on the `W_PCT` scale:

| Rows scored | linear MAE | logit MAE | linear raw range | logit raw range |
|---|---:|---:|:--|:--|
| Chronological holdout (60 team-seasons) | 4.92 W | **4.85 W** | in `[0,1]` | in `[0,1]` |
| Real team lines, current season | 4.9 W | **4.6 W** | `[+0.215, +0.841]` | `[+0.220, +0.814]` |
| Aggregated rosters, conservation ON | **7.4 W** | 7.5 W | `[+0.082, +0.868]` | `[+0.140, +0.836]` |
| Aggregated rosters, conservation OFF | 13.1 W | **12.5 W** | `[−0.048, +0.989]` | `[+0.084, +0.895]` |

Test R² 0.806 → 0.813. Across eight one-season-ahead holdouts the two trade
wins season by season (four each, all within 0.3 wins). This is the expected
result and not a disappointment: the logistic is near-linear across the
`0.12–0.89` band real teams occupy, so on real teams the transform has almost
nothing to do. It earns its place at the extremes — the five-star stress roster
predicted **1.004** (a "200-win season") under the linear model with
conservation on and **−2.609** with it off; both are now 0.881 and 0.000. One
real aggregated team line also came out at −0.048 under the linear model.
*Decision:* **integrated.** The accuracy case is neutral; the correctness case
is that the fictional path is the whole point of the program and it is the path
that produced impossible numbers.

*The cost of the fix, and the guard that pays it.* Bounding the output removes
the tell. A win rate of 2.44 announces itself as nonsense; a clean-looking 82-0
does not, and it is what a saturated logistic returns. So `train_model.py` also
ships `fit_extrapolation_guard()`: Mahalanobis distance from the training teams'
center on standardized features (Ledoit-Wolf shrunk covariance — PACE and POSS
correlate at ~0.99, leaving the raw covariance near-singular), with both
thresholds read off the real teams themselves rather than from a chi-square
table, since multivariate normality is not something this data has to honor.
`edge` is the 99th-percentile training distance (5.78), `limit` is the maximum
(6.36) — the point where extrapolation begins, past which no real team-season
in the sample speaks to the row. A per-feature bounding box alone would not do:
the ordinary five-man rotation used to check this breaks exactly one feature's
range while landing 2.1× out, because what is wrong with it is the
*combination*, which is what a covariance-aware distance sees and a range check
does not.

*Calibrating the threshold.* Walking the guard forward one season at a time
(fit on seasons 1..k, score season k+1; 8 folds, 240 unseen real teams), 4.2% of
genuinely real teams stepped past the prior seasons' maximum, and the worst ever
reached **1.13×** it. So a ratio near 1.0 means "a team shape the league hadn't
produced yet," which happens, and the flag's magnitude — not the flag — is the
signal.

*Evidence — the flag fires on everything, which is itself the finding.* Measured
across 180 real team-seasons:

| Row shape | distance | ratio to `limit` | flagged |
|---|---:|---:|---:|
| Real team-season the guard never saw | ≤ 7.0 | ≤ 1.13× | 4.2% |
| Real team's own top-15 rotation, aggregated | median 10.9 | 1.7× (max 2.4×) | **100%** |
| Real team's own top-5 rotation, aggregated | median 12.0 | 1.9× (max 3.3×) | **100%** |
| Five-star stress roster, conservation ON | 32.5 | 5.1× | 100% |
| Cross-era five-star roster | 40.5 | 6.4× | 100% |
| Five-star stress roster, conservation OFF | 58.4 | 9.2× | 100% |

Every roster that goes through `aggregate_team()` lands outside the real-team
cloud — including a real team's own starting five, whose actual season really
happened. The aggregation step displaces a line off that cloud by itself,
before any fictional-ness enters (rescaling five players to 240 minutes puts
DREB and PFD past any real team's range, among others). *Decision:* keep the
threshold defined by the 360 real teams — that genuinely is where the model's
evidence ends — but report the **ratio** as the headline, with the measured
scale above printed alongside it, and record that the boolean is a constant on
the roster path. A flag that always fires is worth nothing; a ratio that
separates an ordinary rotation (1.9×) from a stacked five (5.1×) by 2.7× is
worth something. This is not a defect the guard introduced, it is a property of
the aggregation that nothing had previously measured.

*Footnote from the rejected 12-feature experiment.* Every ratio above is a
distance in 27-dimensional feature space, and the scale is not portable: guarded
on the 12 rate-and-defence columns instead, the same real 15-man rotations sit
at a median of 1.1× (73% flagged rather than 100%) and the five-star roster at
2.8× rather than 5.1×. Most of the displacement measured above is the rescale to
240 minutes pushing `DREB` and `PFD` past every real team's range, so a feature
set without them looks much less exotic without being any more supported. Worth
knowing before comparing a ratio to one measured elsewhere; the shipped guard is
the 27-feature one.

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
Both now live in `perturbation_tests.py` rather than inline in
`aggregate_team.py` — the tests themselves are unchanged (verified: the full
`python aggregate_team.py` output is byte-identical before and after the move,
and the standalone `python perturbation_tests.py` prints the same section).
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

**5-man lineup DEF_RATING from players' prior-season stats
(`lineup_defense_model.py`).** The fourth and most careful attempt at the
DEF_RATING problem. The three earlier ones (sub-model, personal composite,
awards composite) all either failed or only "worked" by borrowing the real,
current-season, on-court DEF_RATING of the team being predicted. *Approach:*
predict a real 5-man lineup's own observed DEF_RATING from its five players'
**prior-season** individual stats — a 2023-24 lineup described only by its
players' 2022-23 lines, which makes it causally impossible for a feature to
contain the target — with **all on-court team context forbidden from every
season** (player DEF/OFF/NET_RATING, PLUS_MINUS, any on/off split; PIE and
USG_PCT are also excluded as not cleanly "his own"). Features: own box score
(BLK, STL, DREB, OREB, PF, PFD, BLKA per 36), own rebound rates, closest-
defender tracking (opponent FG% and its league-baseline delta), height, weight,
age, MIN, GP, position counts, and prior-season roster continuity — 32 features,
27 after pruning. Possession-weighted throughout.

*Feature construction.* Five players are an unordered set, so features are
pooled, not concatenated by slot. Two things fell out of doing that honestly:
at a fixed group size of five a **sum is exactly 5× the mean**, so including
both would make the design matrix exactly singular (mean only is kept); and
pooling is order-free in exact arithmetic but *not* bit-identical in floating
point, so each lineup's players are sorted by `PLAYER_ID` before aggregating.
`test_permutation_invariance()` shuffles all five slots in every row and asserts
the feature matrix is **byte-identical** — it passes (999 of 1,005 rows actually
reordered at the primary floor). Order statistics (max/min/std of `BLK_36`,
`STL_36`, `DREB_PCT`, height) sit on top of the means, since a mean hides "this
lineup has one elite rim protector."

**Sample-weight scale silently disabled the ridge penalty.** *Found:* RidgeCV
was selecting α = 1000 — exactly the top of `train_model.py`'s
`logspace(-3, 3, 25)` grid — and Ridge and OLS agreed to three decimals. Both
were symptoms of one bug. sklearn's ridge objective is
`Σ wᵢ(yᵢ − xᵢ·β)² + α‖β‖²`, so weights and penalty share a scale; weights were
being passed as **raw season-total possessions** (mean ≈ 550 at the primary
floor), which multiplies the data term by ~550 and therefore divides the
effective penalty by ~550. α = 1000 was really acting like α ≈ 2 — an
essentially unregularized fit, which is why "Ridge ≈ OLS" looked like evidence
of a robust signal and was nothing of the kind. *Fix:* mean-normalize the
weights (relative weighting, and so every possession-weighted metric, is
unchanged) and extend the grid to `logspace(-3, 6, 40)`. *Evidence after the
fix:* α lands at 41–346 across folds and floors — strictly **interior** to the
grid every time — and ridge now actually shrinks: `‖ridge coef‖ / ‖OLS coef‖`
= 0.78 → 0.16 as the possession floor rises and rows get scarcer, exactly the
direction more shrinkage should move. It also changed the ranking: Ridge now
beats OLS at every floor on the headline metric (season-centered R² 0.058 vs
0.003 at floor 250), where before they were indistinguishable. *Standing rule:*
a CV-selected hyperparameter pinned to a grid endpoint is a bug report, not a
result.

**Rolling-origin CV replaced the single 2-season holdout.** *Why:* the old
evaluation trained on 10 of 12 seasons and reported one number from ~160
lineups. With correlated rows (see Limitations) that is one season's weather,
not a measurement — and several conclusions drawn from it did not survive.
*Approach:* expanding chronological window, never shuffled — fold 1 trains
2014-15..2019-20 and tests 2020-21, fold 2 adds 2020-21 to training and tests
2021-22, out to 2025-26; six held-out seasons, ~340–3,200 out-of-fold lineups
depending on floor. 95% CIs are computed **across folds** (t, df = 5), not by
bootstrapping rows: lineups inside a team-season share four of five players, so
a row bootstrap would be anti-conservative.

*Headline metric is the season-centered target.* League-average team DEF_RATING
rose **104.7 → 114.7** across the sample, so a raw-target R² is largely being
charged for a league-wide scoring-environment shift that has nothing to do with
which five players are on the floor (the constant baseline's raw-target bias is
−2.5 to −4.8 points depending on floor). Centering on the season's league
average and scoring back in raw points isolates the lineup question; using it in
production requires forecasting next season's league average separately, which
is a one-number league-level problem the raw model tries to do implicitly and
badly.

*Possession floor, re-picked under the CV folds* (Ridge, season-centered, best
model at every floor):

| POSS floor | lineups (OOF) | R² ceiling | R² mean [95% CI] | share of ceiling | MAE gain vs const | team slope | team R² | team-seasons | team MAE (cal.) vs league avg |
|---:|---:|---:|:--|---:|---:|---:|---:|---:|:--|
| 50  | 6,446 (3,177) | 0.167 | **+0.026 [+0.016, +0.037]** | 0.158 | 0.179 | 0.716 | 0.218 | 179 | 2.24 vs 2.23 (worse) |
| 100 | 2,621 (1,227) | 0.227 | +0.035 [−0.003, +0.073] | 0.153 | 0.169 | 0.712 | 0.224 | 166 | 2.17 vs 2.19 |
| 200 | 1,047 (477)   | 0.285 | +0.058 [−0.005, +0.120] | **0.202** | 0.200 | 0.708 | 0.234 | 142 | 2.05 vs 2.11 |
| 250 |   757 (335)   | 0.330 | +0.058 [−0.006, +0.122] | 0.176 | **0.202** | 0.705 | **0.240** | 133 | **1.96 vs 2.07** |

*Evidence:* the trade is not "more data is better." At 50 possessions ~83% of
the label's variance is sampling noise, and that floor's team-level estimate is
**no better than predicting the league average** (2.24 vs 2.23) — its extra
rows are mostly noise rows. Higher floors win on lineup-level R², on share of
the achievable ceiling, and on the team-level numbers the estimate would
actually be used through. *Decision:* `PRIMARY_POSS_FLOOR = 250`, chosen on the
team-level column; floor 200 is statistically indistinguishable with 42% more
held-out lineups and is the conservative alternative. Two caveats kept in view:
above floor 50 the lineup-level CI **includes zero** (only floor 50's small
effect is separable from zero across folds), and a high floor leaves fewer
team-seasons with any qualifying lineup (133 of 180 vs 179), so its calibration
is fit on a self-selected set of stable rotations.

*Model comparison* (floor 250, six folds, identical features/weights,
season-centered target, MAE in DEF_RATING points):

| Model | R² mean [95% CI] | worst fold | best fold | MAE | pooled-OOF R² |
|-------|:--|---:|---:|---:|---:|
| **Ridge** (α 70–346, interior) | **+0.058 [−0.006, +0.122]** | −0.027 | +0.141 | **4.52** | **+0.096** |
| OLS | +0.003 [−0.111, +0.117] | −0.131 | +0.153 | 4.66 | +0.048 |
| Gradient boosting (defaults) | −0.066 [−0.152, +0.020] | −0.165 | +0.063 | 4.82 | −0.026 |
| Baseline: constant (league avg) | −0.009 | −0.036 | −0.000 | 4.73 | +0.025 |
| Baseline: BLK+STL+DREB sum | −0.005 | −0.034 | +0.020 | 4.75 | +0.028 |

*Evidence:* ridge is the model — the honest ordering only became visible after
the weight fix. The naive blocks+steals+rebounds sum is worth **nothing** over a
constant, a useful negative result about the raw defensive box score. Gradient
boosting is negative on 4 of 6 folds; it had a real chance here (~10× the team
model's rows, and rim protection is where threshold effects would live), was fit
at sklearn defaults with no tuning, and lost anyway — the same way it lost for
the team model.

*Roster continuity: a null, and a retraction.* The single-holdout run had shown
continuity lineups defending ~1.5 points better than their players' prior stats
predict and reshuffled ones ~1.1 points worse (a 2.7-point bias spread), which
looked like a real chemistry effect worth modeling. *Approach:*
`CONTINUITY_PAIR_FRAC` — of the 10 pairs among five players, the fraction who
ended the prior season on the same team. Prior-season membership only: no
minutes-together, no current-season performance, nothing dated after the roster
is known, so it is computable before a season starts and for a roster that has
never played (a cross-era fictional five scores 0.0). *Evidence:* the feature is
a **null** — ridge coefficient −0.073 DEF_RATING points per SD, season-centered
R² 0.058 with it vs 0.057 without. And the effect it was built for largely
**was not there**: under rolling-origin CV the group bias spread is 0.99 points,
not 2.7, and the MAE gap *reverses sign* — mover lineups are predicted **better**
(MAE 4.35 vs 4.72, R² 0.122 vs 0.050), not worse. Adding continuity narrows the
bias spread 0.99 → 0.89; a per-group intercept (a diagnostic refit that uses
current-season team identity and is deliberately not a model feature) gets it to
0.79 and moves the MAE gap by 0.001, so what remains is a small level effect
with **no sign of context memorization** — a model leaning on "this team defends
well" would have been much more accurate on the continuity group, and is the
opposite. *Decision:* kept (it costs nothing and is the right shape for the
question), but recorded as a null rather than the fix it was meant to be, and
the earlier 2.7-point continuity claim is **withdrawn as single-holdout noise**.

*Per-feature sensitivity, then pruning.* `perturbation_impact()` here is the
same diagnostic as `perturbation_tests.py`'s, with each feature moved ±1 SD
instead of by an aggregation MAPE (there is no aggregation step to measure) and
the output in DEF_RATING points rather than wins. Top of the ranking:
`BLKA_36_mean` (0.44 pts/SD), `PCT_PLUSMINUS_mean` (0.35), `BLK_36_min` (0.32),
`BLK_36_max` (0.27), `BLK_36_mean` (0.27), `STL_36_mean` (0.25) — shot-blocking
and closest-defender columns, which is the right shape for a defensive model;
bottom: `DREB_PCT_max`, `MIN_mean`, height min/std, `DREB_36_mean`. Pruning the
three collinear blocks the coefficient table exposed (the offsetting
`OREB_36_mean` / `OREB_PCT_mean` pair; `PCT_PLUSMINUS_mean`, mechanically
`D_FG_PCT − NORMAL_FG_PCT`; the `DREB_PCT` std/max/min triple collapsed to
`DREB_PCT_max`) took 32 features to 27 with Ridge season-centered R² 0.058 →
0.051 → 0.048 → 0.051. *Evidence:* pruning is a **wash to slightly negative** —
every step is far inside the ±0.06 CI, and ridge was already handling the
collinearity (that is what the L2 penalty is for). *Decision:* keep the pruned
27-feature set on parsimony grounds, with the ~0.007 R² cost recorded rather
than presented as an improvement.

*Reliability / calibration (required for integration).* Out-of-fold predictions
from all six held-out seasons, rolled up to team level (possession-weighted),
regressed as true team DEF_RATING on the estimate, 133 held-out team-seasons:

| Model / target | slope | intercept | R² | sd(est) | sd(true) | team MAE raw → calibrated | league-avg baseline |
|---|---:|---:|---:|---:|---:|:--|---:|
| **Ridge, season-centered** | **0.679** | **37.85** | **0.233** | 2.00 | 2.82 | 2.93 → **1.98** | 2.07 |
| OLS, season-centered | 0.483 | 59.40 | 0.202 | 2.62 | 2.82 | 3.10 → 2.01 | 2.07 |
| Ridge, raw target | 0.493 | 59.62 | 0.173 | 2.37 | 2.82 | 5.28 → 2.06 | 2.07 |
| OLS, raw target | 0.389 | 70.61 | 0.151 | 2.81 | 2.82 | 4.79 → 2.09 | 2.07 |

`DRtg_used = 37.85 + 0.679 × DRtg_estimated` is the number a margin-based win
model would have to apply. *Evidence:* the shrinkage is much less severe than
the 0.394 the buggy-ridge single-holdout run reported, and the slope is
remarkably stable across possession floors (0.705–0.716 for the unpruned fit at
every floor), which is what a real coefficient looks like. R² 0.233 of
between-team defensive variance, out of sample, from player stats that predate
the season and contain no team context, is a genuine result — for scale, the
DEF_RATING sub-model rejected earlier in this log managed R² ≈ 0.19 using the
*current* season's box score. *But* R² 0.233 only cuts the residual SD by ~12%,
so at team level the calibrated estimate beats "just predict the league average"
by 1.98 vs 2.07 points of MAE — about 4%. *Decision:* still **not integrated.**
The estimate is real, is now honestly calibrated, and is not yet worth a
feature: 4% of team-level MAE cannot carry the weight of replacing a dropped
column that the win model treats as one of its two dominant drivers. The
slope/intercept above are what a future integration must apply.

*Hustle stats: null under CV.* Contested shots, deflections, defensive box-outs,
charges drawn and loose balls exist only from 2016-17 (2015-16 returns a
147-player, median-1-game partial season that looks like real data — see
`pull_player_defense.py`), so using them costs the three oldest lineup seasons.
On the identical reduced sample and identical folds: Ridge season-centered R²
**0.030 [−0.224, +0.285] with** hustle vs **0.039 [−0.224, +0.303] without**,
MAE 4.60 vs 4.58. *Decision:* **off by default**, kept behind `use_hustle` with
the comparison printed every run. (The single-holdout run had made hustle look
like a 0.23-point MAE gain on the raw target; under CV on the centered target it
is nothing — another number that did not survive the better evaluation.)

*Rookies.* A lineup is dropped if any of its five players lacks a usable prior
season, rather than imputed — imputing a league-average defender would inject a
fabricated feature vector under a real label. This is the single largest sample
cut: at floor 250, 230 of 1,005 lineups (23%) are dropped for a missing prior
season and 18 more for a prior season below `MIN_GP`/`MIN_MPG` (at floor 50 it
is 34%). See Limitations for the resulting bias.

## Limitations & next steps

- **The coefficient signs are wrong and the fix cost more than it bought.** 7 of
  16 priced features fit backwards (`OFF_RATING` −0.877, `AST` −0.536, `TOV`
  +0.550) because collinear features carry offsetting values that cancel only
  while they keep moving together. Real teams always satisfy that; an aggregated
  roster is not guaranteed to. A feature set built to remove the problem was
  tested and reverted for doubling the aggregated-roster error (Test log), so the
  caveat stands **unresolved** and is printed with the coefficients on every run.
  A better answer would improve the two rebound-share approximations first, which
  would raise the ceiling on both feature sets at once.
- **One measured configuration was never fully explored.** Swapping `DREB_PCT`
  for `DREB` beat the reverted 12 on every axis (holdout 3.97 vs 5.08,
  aggregated rosters 10.0 vs 15.2) and beat the shipped 27 on the holdout, losing
  only on aggregated rosters (10.0 vs 7.6). It is a one-line change and the first
  thing to test if anyone picks this up. Three concerns are unmeasured: `DREB` is
  a count, so a cross-era roster carries its era's pace and era adjustment is
  unbuilt; `DREB` takes the 240-minute rescale without the usage pullback; and
  dropping `DREB_PCT` removes the five-centre clip as a visible red flag.

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
  scaled totals). A feature set built to remove the problem was rejected for
  costing more accuracy than it bought (Test log), so the caveat stands
  unresolved and is printed with the coefficients on every run.
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