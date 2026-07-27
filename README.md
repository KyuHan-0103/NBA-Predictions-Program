# NBA Record Prediction Program

Predicts an NBA team's regular-season **win rate** from its team-level statistics
using machine learning.

## Current status

**v1 (working):** Predicts a *real* NBA team's season win rate from that team's
per-game stats, evaluated honestly on seasons the model never trained on.

**Planned (not yet built):** Accept a user-created *fictional* roster of 5–15
players (present or historic), aggregate those players into a team-level
statistical profile, and predict the team's record — eventually accounting for
role/fit, roster depth, and coaching. Cross-era rosters will require era
adjustment. See "Limitations & next steps."

## Features

**Now**
- Pulls team-level stats (Base + Advanced) from the NBA Stats API via `nba_api`
- Cleans and formats the data into one row per team per season
- Trains and compares two models (ridge regression, gradient boosting)
- Evaluates on unseen seasons and reports accuracy in wins

**Planned**
- Accept user-built rosters (5–15 players), with an optional coach
- Adjust user-built rosters to operate without fatigue or injury
- Support historic players via era adjustment
- Roster → team-profile aggregation (usage, minutes, fit)

## Project structure

| File | Purpose |
|------|---------|
| `pull_team_stats.py` | Pulls Base + Advanced team stats, merges on TEAM_ID + SEASON, converts counting stats to per-game, writes `team_season_stats.csv`. Data only — no modeling. |
| `train_model.py` | Loads the CSV, splits by season, trains ridge and gradient boosting on an identical split, prints a comparison table and each model's top drivers. |
| `team_season_stats.csv` | Generated data (git-ignored). |
| `requirements.txt` | Dependencies. |
| `SPEC.md` | Full project specification. |

## Setup & commands

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python pull_team_stats.py         # fetch data -> team_season_stats.csv
python train_model.py             # train + compare models
```

## Configuration

Knobs live at the top of each script:

- `NUM_SEASONS` — how many seasons to pull (`pull_team_stats.py`)
- `N_TEST_SEASONS` — how many recent seasons to hold out for testing (`train_model.py`)
- `ALPHAS` — ridge penalty grid searched by cross-validation

## Methodology

**Target — win rate, not raw wins.** The target is `W_PCT`. Raw win totals aren't
comparable across seasons of different length (e.g. the shortened 2019-20 and
2020-21 seasons), so win *rate* is used and can be multiplied by 82 to report a
projected record.

**Features — per-game and rate stats.** Season-total counting stats (points,
rebounds, etc.) are divided by games played so a 72-game season and an 82-game
season sit on the same scale. Stats that are already rates (shooting
percentages, efficiency ratings, pace) are left as-is.

**Leakage control.** Any column that directly contributes to the target is removed, 
and redundant columns are also removed: wins/losses and their ranks, `W_PCT`-derived fields, 
`PLUS_MINUS`, and `NET_RATING`. These would let a model reconstruct the target instead 
of learning from basketball stats. Stats like `FT_PCT` are removed as they can be constructed from `FTA` and `FTM`.

**Evaluation — chronological holdout.** The model trains on older seasons and is
tested on the most recent seasons it has never seen. This mirrors real
forecasting (predicting a season before it happens) and avoids the mild leakage
of a random split, where the same team's adjacent seasons could straddle the
train/test line.

## Results

**Production model:** ridge regression trained on the modern era (2014-15 onward).

- Predicts held-out-season win rate with **MAE ≈ 0.032 → about 2.6 wins over an
  82-game season.**
- **Test R² ≈ 0.94**, with essentially no gap between train and test scores
  (no overfitting).
- **Dominant drivers:** offensive and defensive efficiency — a finding both
  models independently agree on.

**Model comparison & data-volume experiment.** Ridge beat gradient boosting at
every dataset size tested. Gradient boosting overfit heavily on  small data;
its train/test gap narrowed as seasons were added but never overtook ridge:

| Training rows | GB train R² | GB test R² | GB gap |
|---------------|-------------|------------|--------|
| 300 (10 seasons) | 0.986 | 0.916 | 0.070 |
| 360 (12 seasons) | 0.981 | 0.919 | 0.062 |
| 540 (18 seasons) | 0.973 | 0.931 | 0.042 |

**Conclusion:** season-level prediction is inherently small-data — only 30
team-seasons exist per year — which favors a simple, regularized linear model.
More data helps a complex model generalize, but not enough to justify it here.

## Limitations & next steps

- **Small data.** Roughly 30 rows per season caps how complex a model can
  usefully be.
- **Era mixing.** Including pre-2014 seasons mixes a different style of
  basketball into the training data and degrades comparability; the production
  model deliberately uses only the modern era. (The pre-2014 pull is retained as
  a data-volume experiment, not as the shipping model.)
- **No ground truth for the end goal.** Fictional cross-era teams never played,
  so their predictions are informed estimates that cannot be directly verified.
  The model can only be *graded* on real teams.
- **Planned builds:** player-level data → roster-to-team aggregation (validated
  against real teams first) → era adjustment for historic players.
- **DEF_RATING has no honest analogue for a fictional roster.** OFF_RATING can
be reconstructed from an aggregated box score (points ÷ possessions);
DEF_RATING can't, since it's opponent-dependent — nothing in a roster's own
production determines how many points its opponents scored. Three handlings
were tested end-to-end through `aggregate_team.py` (aggregate every real
team's actual roster, score it with the win-rate model, compare to actual
wins):

- *Minutes-weighted average of players' own real DEF_RATING* scored best
  (~6.5 wins MAE) — but this number is an artifact of validating on real
  teams. Each player's DEF_RATING was earned inside that team's scheme with
  those teammates, so rebuilding a team from its own players nearly
  reconstructs the original value. On a genuinely fictional or cross-era
  roster, whose players earned their ratings elsewhere, that circularity is
  gone and there's no reason the average would beat dropping the column.
- *A composite built from personal defensive box-score stats* (steals,
  blocks, defensive rebounds, fouls) did not meaningfully beat dropping
  DEF_RATING (~7.57 vs. ~7.60 wins MAE). Its small edge on real-team
  *training* traced to collapsing four collinear columns into one — a mild
  regularization effect, not recovered defensive signal — and it vanished
  through aggregation.

`aggregate_team.py` therefore **drops DEF_RATING** on the fictional-roster
path. The ~7.6-win aggregated MAE is the honest error floor: the part of
defense that lives in scheme and opponent behavior can't be manufactured
from a roster's own stats by any method tested, so it's stated as a
limitation rather than papered over. (The real-team analysis in
`train_model.py` is unchanged and still uses real DEF_RATING; only the
fictional-roster path drops it.)