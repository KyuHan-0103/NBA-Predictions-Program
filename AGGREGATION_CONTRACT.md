## Aggregation contract (aggregate_team)

Precondition (input):
- 5–15 players, eligibility filter already applied (GP/MIN thresholds).
- Player stats already per-game (season totals ÷ GP), not raw totals.

Postcondition (output):
- Exactly one row, columns == train_model.py feature_cols, in that order.
  Nothing extra, nothing missing (the existing missing-feature raise stays).
- No NaN/inf. Every ratio (AST/TOV, PTS/POSS, ...) has a divide-by-zero guard.
- Each feature matches the UNITS AND DEFINITION of the team CSV column it fills:
  per-game vs per-100 vs fraction vs percent must agree. (TM_TOV_PCT is the known
  trap — confirm the team CSV stores it the same way this function does.)

Invariants (modeling rules, not just code):
- Ball is zero-sum: the five on-court players' usage/volume cannot all be
  preserved at full rate simultaneously (see usage-conservation work).
- No leakage columns ever appear (W, L, W_PCT, PLUS_MINUS, NET_RATING and the
  on/off ratings are excluded upstream; the output must not reintroduce them).
- OFF_RATING is reconstructable from the team's own box score; DEF_RATING and
  rebound shares are APPROXIMATIONS — label them as low-confidence, don't treat
  them as ground truth.
- The output feeds a model fitted on logit(W_PCT), so a predicted win rate is
  always inside (0, 1) — that is a property of the transform, NOT a sign the
  aggregated line is realistic. Every aggregated roster, real ones included,
  falls outside the training teams' feature-space region; report the
  extrapolation guard's ratio with any prediction (train_model.extrapolation_report).