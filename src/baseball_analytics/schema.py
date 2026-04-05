WAREHOUSE_DDL = """
CREATE TABLE IF NOT EXISTS dim_team (
    team_key VARCHAR,
    team_id VARCHAR,
    franchise_id VARCHAR,
    team_name VARCHAR,
    league_id VARCHAR,
    PRIMARY KEY(team_key)
);

CREATE TABLE IF NOT EXISTS dim_season (
    season_key INTEGER,
    year_id INTEGER,
    PRIMARY KEY(season_key)
);

CREATE TABLE IF NOT EXISTS fact_team_season (
    team_key VARCHAR,
    season_key INTEGER,
    wins INTEGER,
    losses INTEGER,
    games INTEGER,
    runs_scored INTEGER,
    runs_allowed INTEGER,
    strikeouts INTEGER,
    attendance DOUBLE,
    run_diff INTEGER,
    pythag_wins DOUBLE,
    payroll DOUBLE,
    max_salary DOUBLE,
    median_salary DOUBLE,
    top_1_salary_share DOUBLE,
    top_3_salary_share DOUBLE,
    top_5_salary_share DOUBLE,
    gini_salary DOUBLE,
    payroll_per_win DOUBLE,
    wins_per_10m DOUBLE,
    run_diff_per_10m DOUBLE,
    PRIMARY KEY(team_key, season_key)
);

CREATE TABLE IF NOT EXISTS fact_salary (
    season_key INTEGER,
    team_id VARCHAR,
    player_id VARCHAR,
    salary DOUBLE
);
"""
