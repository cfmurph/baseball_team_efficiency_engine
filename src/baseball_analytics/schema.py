WAREHOUSE_DDL = """
-- ----------------------------------------------------------------
-- Dimension: Team
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE dim_team (
    team_key    VARCHAR PRIMARY KEY,
    team_id     VARCHAR,
    franchise_id VARCHAR,
    team_name   VARCHAR,
    league_id   VARCHAR
);

-- ----------------------------------------------------------------
-- Dimension: Season
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE dim_season (
    season_key  INTEGER PRIMARY KEY,
    year_id     INTEGER
);

-- ----------------------------------------------------------------
-- Dimension: Player
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE dim_player (
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
CREATE OR REPLACE TABLE fact_salary (
    season_key  INTEGER,
    team_id     VARCHAR,
    player_id   VARCHAR,
    salary      DOUBLE,
    PRIMARY KEY (season_key, team_id, player_id)
);

-- ----------------------------------------------------------------
-- Fact: Player Season (batting + pitching + WAR)
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE fact_player_season (
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
CREATE OR REPLACE TABLE fact_team_season (
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

-- ----------------------------------------------------------------
-- Sportradar: Team ID crosswalk  (SR GUID ↔ Lahman teamID)
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE dim_sportradar_team_map (
    sr_team_id      VARCHAR PRIMARY KEY,
    sr_abbr         VARCHAR,
    sr_market       VARCHAR,
    sr_name         VARCHAR,
    lahman_team_id  VARCHAR,
    lahman_franch_id VARCHAR
);

-- ----------------------------------------------------------------
-- Sportradar: Player season stats (real WAR, wOBA, wRC+, FIP, ERA-)
-- Grain: player × year × team (REG season only)
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE fact_sr_player_season (
    sr_player_id    VARCHAR,
    sr_team_id      VARCHAR,
    season_year     INTEGER,

    -- Identity
    full_name       VARCHAR,
    position        VARCHAR,
    primary_position VARCHAR,
    jersey_number   INTEGER,

    -- Hitting
    pa              DOUBLE,
    ab              DOUBLE,
    hits            DOUBLE,
    doubles         DOUBLE,
    triples         DOUBLE,
    hr              DOUBLE,
    rbi             DOUBLE,
    bb              DOUBLE,
    ibb             DOUBLE,
    hbp             DOUBLE,
    sb              DOUBLE,
    avg             DOUBLE,
    obp             DOUBLE,
    slg             DOUBLE,
    ops             DOUBLE,
    woba            DOUBLE,
    wraa            DOUBLE,
    wrc             DOUBLE,
    wrc_plus        DOUBLE,
    war             DOUBLE,
    bwar            DOUBLE,
    brwar           DOUBLE,
    fwar            DOUBLE,

    -- Pitching
    ip              DOUBLE,
    era             DOUBLE,
    era_minus       DOUBLE,
    fip             DOUBLE,
    whip            DOUBLE,
    k9              DOUBLE,
    bb9             DOUBLE,
    hr9             DOUBLE,
    kbb             DOUBLE,
    p_war           DOUBLE,

    PRIMARY KEY (sr_player_id, season_year, sr_team_id)
);

-- ----------------------------------------------------------------
-- Sportradar: Transactions log
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE fact_sr_transactions (
    transaction_id      VARCHAR PRIMARY KEY,
    effective_date      DATE,
    last_modified       TIMESTAMP,
    transaction_type    VARCHAR,
    transaction_code    VARCHAR,
    description         VARCHAR,
    sr_player_id        VARCHAR,
    player_name         VARCHAR,
    from_team_abbr      VARCHAR,
    to_team_abbr        VARCHAR,
    from_sr_team_id     VARCHAR,
    to_sr_team_id       VARCHAR
);

-- ----------------------------------------------------------------
-- Sportradar: Injuries
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE fact_sr_injuries (
    sr_player_id    VARCHAR,
    player_name     VARCHAR,
    sr_team_id      VARCHAR,
    team_abbr       VARCHAR,
    injury_desc     VARCHAR,
    injury_status   VARCHAR,
    start_date      DATE,
    end_date        DATE,
    fetched_at      TIMESTAMP,
    PRIMARY KEY (sr_player_id, start_date)
);
"""
