WAREHOUSE_DDL = """
-- ----------------------------------------------------------------
-- Dimension: Team
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_team (
    team_key    VARCHAR PRIMARY KEY,
    team_id     VARCHAR,
    franchise_id VARCHAR,
    team_name   VARCHAR,
    league_id   VARCHAR
);

-- ----------------------------------------------------------------
-- Dimension: Season
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_season (
    season_key  INTEGER PRIMARY KEY,
    year_id     INTEGER
);

-- ----------------------------------------------------------------
-- Dimension: Player
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_player (
    player_id       VARCHAR PRIMARY KEY,
    name_first      VARCHAR,
    name_last       VARCHAR,
    name_full       VARCHAR,
    birth_year      INTEGER,
    birth_country   VARCHAR,
    throws          VARCHAR,
    bats            VARCHAR
);

-- ----------------------------------------------------------------
-- Fact: Salary (player-season-team)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_salary (
    season_key  INTEGER,
    team_id     VARCHAR,
    player_id   VARCHAR,
    salary      DOUBLE,
    PRIMARY KEY (season_key, team_id, player_id)
);

-- ----------------------------------------------------------------
-- Fact: Player Season (batting + pitching + WAR)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_player_season (
    player_id       VARCHAR,
    season_key      INTEGER,
    team_id         VARCHAR,
    player_type     VARCHAR,     -- 'batter' | 'pitcher' | 'both'

    -- Batting
    pa              DOUBLE,
    hr              DOUBLE,
    bb              DOUBLE,
    woba            DOUBLE,
    batting_war     DOUBLE,

    -- Pitching
    ip              DOUBLE,
    fip             DOUBLE,
    era             DOUBLE,
    pitching_war    DOUBLE,

    -- Combined
    player_war      DOUBLE,
    salary          DOUBLE,
    surplus_value   DOUBLE,
    contract_label  VARCHAR,

    PRIMARY KEY (player_id, season_key, team_id)
);

-- ----------------------------------------------------------------
-- Fact: Team Season (team-level aggregated metrics)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_team_season (
    team_key            VARCHAR,
    season_key          INTEGER,

    -- On-field
    wins                INTEGER,
    losses              INTEGER,
    games               INTEGER,
    runs_scored         INTEGER,
    runs_allowed        INTEGER,
    strikeouts          INTEGER,
    attendance          DOUBLE,
    run_diff            INTEGER,
    pythag_wins         DOUBLE,
    pythag_gap          DOUBLE,

    -- BaseRuns
    base_runs           DOUBLE,
    base_runs_gap       DOUBLE,

    -- WAR
    team_batting_war    DOUBLE,
    team_pitching_war   DOUBLE,
    team_total_war      DOUBLE,
    war_win_gap         DOUBLE,

    -- Payroll
    payroll             DOUBLE,
    max_salary          DOUBLE,
    median_salary       DOUBLE,

    -- Salary concentration
    top_1_salary_share  DOUBLE,
    top_3_salary_share  DOUBLE,
    top_5_salary_share  DOUBLE,
    gini_salary         DOUBLE,
    dead_money_share    DOUBLE,

    -- Efficiency
    payroll_per_win     DOUBLE,
    wins_per_10m        DOUBLE,
    run_diff_per_10m    DOUBLE,
    cost_per_war        DOUBLE,
    war_per_1m          DOUBLE,
    surplus_value       DOUBLE,

    -- Window
    window_phase        VARCHAR,

    PRIMARY KEY (team_key, season_key)
);
"""
