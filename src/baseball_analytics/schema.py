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

    -- Combined (effective: BR rWAR when present, else Lahman approx)
    player_war      DOUBLE,
    war_source      VARCHAR,     -- 'real' | 'approx'
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

    -- WAR (rolled up from effective player WAR)
    team_batting_war    DOUBLE,
    team_pitching_war   DOUBLE,
    team_total_war      DOUBLE,
    war_source          VARCHAR,     -- 'real' | 'approx' | 'mixed'
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

-- ----------------------------------------------------------------
-- MLB Stats API: Team ID crosswalk (MLB id ↔ Lahman teamID)
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE dim_mlb_team_map (
    mlb_team_id      INTEGER PRIMARY KEY,
    mlb_abbr         VARCHAR,
    mlb_name         VARCHAR,
    league_id        INTEGER,
    lahman_team_id   VARCHAR,
    lahman_franch_id VARCHAR
);

-- ----------------------------------------------------------------
-- MLB Stats API: Player ID crosswalk (MLB id ↔ Lahman playerID)
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE dim_mlb_player_map (
    mlb_player_id    INTEGER PRIMARY KEY,
    lahman_player_id VARCHAR,
    player_name      VARCHAR
);

-- ----------------------------------------------------------------
-- MLB Stats API: Team season (standings + hitting/pitching)
-- No WAR columns — BR rWAR remains the WAR source of truth.
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE fact_mlb_team_season (
    mlb_team_id      INTEGER,
    season_year      INTEGER,
    lahman_team_id   VARCHAR,
    team_name        VARCHAR,
    wins             INTEGER,
    losses           INTEGER,
    games            INTEGER,
    runs_scored      INTEGER,
    runs_allowed     INTEGER,
    run_diff         INTEGER,
    winning_pct      DOUBLE,
    batting_hits     INTEGER,
    batting_hr       INTEGER,
    batting_bb       INTEGER,
    batting_so       INTEGER,
    avg              DOUBLE,
    obp              DOUBLE,
    slg              DOUBLE,
    ops              DOUBLE,
    ip               DOUBLE,
    era              DOUBLE,
    whip             DOUBLE,
    pitching_so      INTEGER,
    pitching_bb      INTEGER,
    as_of_date       VARCHAR,
    PRIMARY KEY (mlb_team_id, season_year)
);

-- ----------------------------------------------------------------
-- MLB Stats API: Player season (hitting + pitching)
-- Grain: player × year × team. No WAR columns.
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE fact_mlb_player_season (
    mlb_player_id    INTEGER,
    season_year      INTEGER,
    mlb_team_id      INTEGER,
    lahman_player_id VARCHAR,
    lahman_team_id   VARCHAR,
    player_name      VARCHAR,
    player_type      VARCHAR,
    games            INTEGER,
    pa               DOUBLE,
    ab               DOUBLE,
    hits             DOUBLE,
    hr               DOUBLE,
    bb               DOUBLE,
    so               DOUBLE,
    avg              DOUBLE,
    obp              DOUBLE,
    slg              DOUBLE,
    ops              DOUBLE,
    ip               DOUBLE,
    era              DOUBLE,
    whip             DOUBLE,
    pitching_so      DOUBLE,
    pitching_bb      DOUBLE,
    as_of_date       VARCHAR,
    PRIMARY KEY (mlb_player_id, season_year, mlb_team_id)
);

-- ----------------------------------------------------------------
-- MLB Stats API: Games (schedule + scores)
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE fact_mlb_game (
    game_pk              INTEGER PRIMARY KEY,
    game_date            DATE,
    season_year          INTEGER,
    status               VARCHAR,
    venue_name           VARCHAR,
    home_mlb_team_id     INTEGER,
    away_mlb_team_id     INTEGER,
    home_lahman_team_id  VARCHAR,
    away_lahman_team_id  VARCHAR,
    home_score           INTEGER,
    away_score           INTEGER,
    home_wins            INTEGER,
    home_losses          INTEGER,
    away_wins            INTEGER,
    away_losses          INTEGER,
    as_of_date           VARCHAR
);

-- ----------------------------------------------------------------
-- Phase 0 spine (schema v0.1 LOCKED by Cole 2026-08-23 / #128)
-- Internal PKs are UUID5; live joins go through external_id_alias.
-- Do NOT add fantasy_*_stat or scout_*_stat forks.
-- ----------------------------------------------------------------
CREATE OR REPLACE TABLE player (
    player_id        VARCHAR PRIMARY KEY,
    sdio_player_id   INTEGER,
    display_name     VARCHAR,
    first_name       VARCHAR,
    last_name        VARCHAR,
    position         VARCHAR,
    bats             VARCHAR,
    throws           VARCHAR,
    team_id          VARCHAR,
    source           VARCHAR,
    source_endpoint  VARCHAR,
    computed_at      VARCHAR,
    as_of            VARCHAR,
    run_id           VARCHAR,
    is_approx        BOOLEAN
);

CREATE OR REPLACE TABLE team (
    team_id          VARCHAR PRIMARY KEY,
    sdio_team_id     INTEGER,
    sdio_abbr        VARCHAR,
    city             VARCHAR,
    team_name        VARCHAR,
    league           VARCHAR,
    division         VARCHAR,
    source           VARCHAR,
    source_endpoint  VARCHAR,
    computed_at      VARCHAR,
    as_of            VARCHAR,
    run_id           VARCHAR,
    is_approx        BOOLEAN
);

CREATE OR REPLACE TABLE game (
    game_id              VARCHAR PRIMARY KEY,
    sdio_game_id         INTEGER,
    game_date            DATE,
    season               INTEGER,
    status               VARCHAR,
    home_team_id         VARCHAR,
    away_team_id         VARCHAR,
    home_score           INTEGER,
    away_score           INTEGER,
    source               VARCHAR,
    source_endpoint      VARCHAR,
    computed_at          VARCHAR,
    as_of                VARCHAR,
    run_id               VARCHAR,
    is_approx            BOOLEAN
);

CREATE OR REPLACE TABLE external_id_alias (
    alias_id         VARCHAR PRIMARY KEY,
    entity_type      VARCHAR,     -- player | team | game
    internal_id      VARCHAR,
    system           VARCHAR,     -- sportsdataio | mlb | bbref | fangraphs | lahman
    external_id      VARCHAR,
    is_primary       BOOLEAN,
    source           VARCHAR,
    source_endpoint  VARCHAR,
    computed_at      VARCHAR,
    as_of            VARCHAR,
    run_id           VARCHAR,
    is_approx        BOOLEAN,
    UNIQUE (system, entity_type, external_id)
);

CREATE OR REPLACE TABLE player_game_stat (
    player_id        VARCHAR,
    game_id          VARCHAR,
    team_id          VARCHAR,
    game_date        DATE,
    season           INTEGER,
    position         VARCHAR,
    started          INTEGER,
    pa               DOUBLE,
    ab               DOUBLE,
    runs             DOUBLE,
    hits             DOUBLE,
    doubles          DOUBLE,
    triples          DOUBLE,
    hr               DOUBLE,
    rbi              DOUBLE,
    bb               DOUBLE,
    so               DOUBLE,
    sb               DOUBLE,
    hbp              DOUBLE,
    cs               DOUBLE,
    sh               DOUBLE,
    sf               DOUBLE,
    gidp             DOUBLE,
    ibb              DOUBLE,
    lob              DOUBLE,
    roe              DOUBLE,
    gsh              DOUBLE,
    singles          DOUBLE,
    tb               DOUBLE,
    go               DOUBLE,
    ao               DOUBLE,
    ip               DOUBLE,
    er               DOUBLE,
    era              DOUBLE,
    whip             DOUBLE,
    pitching_so      DOUBLE,
    pitching_bb      DOUBLE,
    pitching_hits    DOUBLE,
    pitching_hr      DOUBLE,
    pitching_r       DOUBLE,
    games_started    INTEGER,
    wins             DOUBLE,
    losses           DOUBLE,
    saves            DOUBLE,
    cg               DOUBLE,
    sho              DOUBLE,
    hld              DOUBLE,
    bs               DOUBLE,
    qs               DOUBLE,
    gf               DOUBLE,
    bk               DOUBLE,
    wp               DOUBLE,
    np               DOUBLE,
    pk               DOUBLE,
    ir               DOUBLE,
    bf               DOUBLE,
    pitching_go      DOUBLE,
    pitching_ao      DOUBLE,
    pitching_hbp     DOUBLE,
    pitching_ibb     DOUBLE,
    woba             DOUBLE,
    fip              DOUBLE,
    iso              DOUBLE,
    babip            DOUBLE,
    -- Pass-through of landed SDIO fielding counting (not DRS/OAA/UZR; not v0.2).
    putouts          DOUBLE,
    assists          DOUBLE,
    errors           DOUBLE,
    double_plays     DOUBLE,
    passed_balls     DOUBLE,
    ofa              DOUBLE,
    fielding_cs      DOUBLE,
    fielding_sb      DOUBLE,
    tp               DOUBLE,
    source           VARCHAR,
    source_endpoint  VARCHAR,
    computed_at      VARCHAR,
    as_of            VARCHAR,
    run_id           VARCHAR,
    is_approx        BOOLEAN,
    PRIMARY KEY (player_id, game_id)
);

CREATE OR REPLACE TABLE player_season_stat (
    player_id        VARCHAR,
    season           INTEGER,
    team_id          VARCHAR,
    games            INTEGER,
    pa               DOUBLE,
    ab               DOUBLE,
    hits             DOUBLE,
    hr               DOUBLE,
    bb               DOUBLE,
    so               DOUBLE,
    rbi              DOUBLE,
    sb               DOUBLE,
    runs             DOUBLE,
    doubles          DOUBLE,
    triples          DOUBLE,
    hbp              DOUBLE,
    cs               DOUBLE,
    sh               DOUBLE,
    sf               DOUBLE,
    gidp             DOUBLE,
    ibb              DOUBLE,
    lob              DOUBLE,
    roe              DOUBLE,
    gsh              DOUBLE,
    singles          DOUBLE,
    tb               DOUBLE,
    go               DOUBLE,
    ao               DOUBLE,
    ip               DOUBLE,
    era              DOUBLE,
    whip             DOUBLE,
    er               DOUBLE,
    pitching_so      DOUBLE,
    pitching_bb      DOUBLE,
    pitching_hits    DOUBLE,
    pitching_hr      DOUBLE,
    pitching_r       DOUBLE,
    games_started    INTEGER,
    wins             DOUBLE,
    losses           DOUBLE,
    saves            DOUBLE,
    cg               DOUBLE,
    sho              DOUBLE,
    hld              DOUBLE,
    bs               DOUBLE,
    qs               DOUBLE,
    gf               DOUBLE,
    bk               DOUBLE,
    wp               DOUBLE,
    np               DOUBLE,
    pk               DOUBLE,
    ir               DOUBLE,
    bf               DOUBLE,
    pitching_go      DOUBLE,
    pitching_ao      DOUBLE,
    pitching_hbp     DOUBLE,
    pitching_ibb     DOUBLE,
    woba             DOUBLE,
    fip              DOUBLE,
    iso              DOUBLE,
    babip            DOUBLE,
    -- Pass-through of landed SDIO fielding counting (not DRS/OAA/UZR; not v0.2).
    putouts          DOUBLE,
    assists          DOUBLE,
    errors           DOUBLE,
    double_plays     DOUBLE,
    passed_balls     DOUBLE,
    ofa              DOUBLE,
    fielding_cs      DOUBLE,
    fielding_sb      DOUBLE,
    tp               DOUBLE,
    source           VARCHAR,
    source_endpoint  VARCHAR,
    computed_at      VARCHAR,
    as_of            VARCHAR,
    run_id           VARCHAR,
    is_approx        BOOLEAN,
    PRIMARY KEY (player_id, season, team_id)
);
"""
