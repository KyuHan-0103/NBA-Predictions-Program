OVERVIEW:
Similar to the popular 82-0 games found online, this program aims to predict a fictional NBA team's success by guessing its record. Instead of solely using statistics, this program also aims to incorporate intangibles such as style of play, leadership, playstyle and more while still keeping the spirit of fantasy where players can cross through eras without injuries or aging.

GOALS:
This program only aims to predict a fictional team's win loss record in today's NBA (only comparing to current teams).
This program does not aim to be able to directly compare two teams
This program takes in 5 - 15 NBA players, present or historic, and returns the predicted amount of wins the players would get in an 82 game season.
The user can optionally choose a coach for their team.
Coaches known for/statistically proven to be able to utilize any and all different players will be utilize player's to their fullest capacity.
This program does not consider injuries, physical exhaustion, mental exhaustian or anything else. Thus, a 5 man roster will not be penalized for only having 5 players.

REQUIREMENTS:

ARCHITECTURE:
Program recieves and formats data from nba api
Ridge model trains on real team's data
Model receives user inputted team
Stores and consider all player's stats
Adjusts stats based on usage, intabigles, roster depth, coaching
Returns predicted record

Roster input (predict_fictional_roster.py, backed by pull_player_seasons_all.py):
User builds a roster interactively, one player at a time, by name.
Each player is looked up as a specific (player, season) pair, not just a
person -- the same player from two different seasons are different, separately
selectable entries (e.g. 2019-20 Stephen Curry vs. 2022-23 Stephen Curry).
For each player found, the user picks either:
  - "Prime": the statistically best season of that player's career, defined
    as the eligible season (see CONSTRAINTS/ASSUMPTIONS) with the highest PIE
    (Player Impact Estimate, an NBA-computed single-number box-score summary).
  - A specific season, entered as e.g. "2019-20".
Once accepted, the roster is passed to aggregate_team.py's aggregation
contract and scored by the same ridge model trained in train_model.py.

API:
nba_api's LeagueDashPlayerStats endpoint (Base + Advanced measure types,
Regular Season, per player per season) is the source for individual player
stats, mirroring the team-level pull in pull_team_stats.py.

EDGE CASES/ERRORS:
The roster-input tool rejects a player pick and re-prompts for another player
(never silently drops or substitutes one) when:
  - the entered name matches no player in the data ("player not found").
  - the entered season isn't formatted as a valid season label, or is outside
    the range this project has data for ("season not available").
  - the named player has no row for the entered season ("player didn't play
    that season").
  - the matched (player, season) row doesn't meet the eligibility filter (GP/MIN
    thresholds) or contains unusable data (NaN, infinite, or all-zero advanced
    stats) -- see CONSTRAINTS/ASSUMPTIONS.
  - the same (player, season) pair is picked twice for one roster.
  - a name matches more than one player and the user doesn't disambiguate.
The tool also rejects trying to finish with fewer than 5 players, and stops
accepting new players once the roster reaches 15.

CONSTRAINTS/ASSUMPTIONS:
User inputs a minimum of 5 players and a maximum of 15 players.
Players are selectable from the 1996-97 season onward -- confirmed empirically
as the first season stats.nba.com's Advanced player data exists (earlier
seasons return zero rows, not zeroed/garbage data).
A player-season only qualifies for roster input (and for "Prime" season
selection) if it clears aggregate_team.py's own eligibility filter (GP >= 5,
MIN >= 6 per game) and its required advanced columns are finite and not all
zero -- the same precondition aggregate_team.py's aggregation contract already
requires of its input.
"Prime" season is a heuristic (highest PIE among eligible seasons), not a
subjective or all-things-considered "best season" judgment -- see README
Limitations.