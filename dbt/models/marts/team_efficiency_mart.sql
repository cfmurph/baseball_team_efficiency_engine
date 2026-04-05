select
  s.year_id,
  t.team_name,
  t.team_id,
  t.franchise_id,
  t.league_id,
  f.wins,
  f.losses,
  f.games,
  f.runs_scored,
  f.runs_allowed,
  f.run_diff,
  f.pythag_wins,
  f.payroll,
  f.max_salary,
  f.median_salary,
  f.top_1_salary_share,
  f.top_3_salary_share,
  f.top_5_salary_share,
  f.gini_salary,
  f.payroll_per_win,
  f.wins_per_10m,
  f.run_diff_per_10m,
  f.wins - f.pythag_wins as pythag_gap
from {{ ref('stg_team_season') }} f
join dim_team t using (team_key)
join dim_season s using (season_key)
