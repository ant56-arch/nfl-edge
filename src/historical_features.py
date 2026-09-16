"""
historical_features.py
Walk-forward ("as of") feature engine shared by fit_model.py and backtest.py.

THE LEAKAGE PROBLEM: team_stats.csv (built by build_features.py) is a single
end-of-data snapshot - every team's "current form" number already includes
every game we've downloaded, including ones we'd want to backtest a
prediction against. Using that snapshot to "predict" a past game is cheating:
the model would be using information from the future (and from the game
itself) to predict that same game.

THE FIX: for every team, walk through its games in chronological order and
compute each metric using ONLY strictly prior games - exactly what a
predictor standing right before kickoff could have known. This reproduces
build_features.py's exponential-decay recency weighting exactly, just
evaluated at every point in time instead of only at the very end.

The recursive form below is mathematically identical to recomputing
build_features.recency_weight() from scratch at every cutoff, but does it in
one O(n) pass per team instead of O(n^2):
  A_i = weighted sum of prior games' values (most recent prior game has
        weight 1, one before that r, two before that r^2, ...)
  W_i = the matching weighted count (for normalizing to an average)
  A_(i+1) = r*A_i + value_i     (decay everything, then add the new game)
  W_(i+1) = r*W_i + 1
where r = 0.5 ** (1 / half_life).
"""

import pandas as pd
import numpy as np

from build_features import RECENCY_HALF_LIFE_GAMES

def compute_walkforward_features(game_stats, metric_cols, half_life=RECENCY_HALF_LIFE_GAMES, group_col="team"):
    """
    game_stats: one row per (season, week, <group_col>) with raw per-game
    metric_cols (as produced by build_team_game_stats / build_team_defense_game_stats,
    or build_player_game_stats for group_col="player_id").

    Returns game_stats with two new columns per metric:
      f"{col}_asof"             - recency-weighted "current form" using only
                                   games strictly before this one
      f"{col}_asof_season_avg"  - unweighted season-to-date average, reset at
                                   each season boundary, using only games
                                   strictly before this one
    Both are NaN for an entity's very first game in the dataset (no prior data).
    """
    r = 0.5 ** (1 / half_life)
    out_frames = []

    for key, g in game_stats.groupby(group_col, sort=False):
        g = g.sort_values(["season", "week"]).reset_index(drop=True)
        n = len(g)

        asof = {c: np.full(n, np.nan) for c in metric_cols}
        asof_season = {c: np.full(n, np.nan) for c in metric_cols}

        A = {c: 0.0 for c in metric_cols}
        W = {c: 0.0 for c in metric_cols}
        sA = {c: 0.0 for c in metric_cols}
        sN = {c: 0 for c in metric_cols}
        current_season = None

        for i in range(n):
            season_i = g.loc[i, "season"]
            if season_i != current_season:
                current_season = season_i
                sA = {c: 0.0 for c in metric_cols}
                sN = {c: 0 for c in metric_cols}

            for c in metric_cols:
                asof[c][i] = A[c] / W[c] if W[c] > 0 else np.nan
                asof_season[c][i] = sA[c] / sN[c] if sN[c] > 0 else np.nan

                v = g.loc[i, c]
                if pd.notna(v):
                    A[c] = r * A[c] + v
                    W[c] = r * W[c] + 1
                    sA[c] += v
                    sN[c] += 1
                else:
                    A[c] = r * A[c]
                    W[c] = r * W[c]

        for c in metric_cols:
            g[f"{c}_asof"] = asof[c]
            g[f"{c}_asof_season_avg"] = asof_season[c]

        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)
