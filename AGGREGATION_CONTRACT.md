## Aggregation contract (aggregate_team)

Precondition (input):
- 5–15 players, eligibility filter already applied (GP/MIN thresholds).
- Player stats already per-game (season totals ÷ GP), not raw totals.

Postcondition (output):
- Exactly one row, columns == train_model.py feature_cols, in that order.
  Nothing extra, nothing missing (the existing missing-feature raise stays).
- No NaN/inf. Every ratio (AST/TOV, PTS/POSS, ...) has a divide-by-zero guard.
- Every *_PCT output must be a fraction in [0, 1]. Enforced two ways, because
  the two kinds of *_PCT column fail for different reasons:
    * CLIPPABLE_PCT_COLS (OREB_PCT, DREB_PCT) are opponent-dependent
      approximations with no ceiling built in — five bigs claim ~148% of the
      opponent's misses. These are CLIPPED into range, and every clip that
      fires is recorded in `result.attrs["pct_clipped"]` as {col: pre-clip
      value}, so a caller can tell a clipped line from a clean one.
    * Any other *_PCT (EFG_PCT, TM_TOV_PCT) is recomputed from the roster's own
      totals and cannot leave [0, 1] unless a formula or its units are wrong.
      Those RAISE (_check_pct_range). Do not add a column to the clip list to
      make a failure go away — that converts a bug into a plausible number.
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
- Report a prediction with its interval (train_model.interval_report), and only
  with the two components it covers named: model error on held-out real teams,
  and 82-game outcome randomness. It does NOT cover the dropped DEF_RATING or
  the roster's extrapolation ratio, and neither is quantifiable from this data.
  A narrow interval that omits its largest term is worse than no interval.