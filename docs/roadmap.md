# Expansion Roadmap

## Phase 1: Stronger team efficiency foundation
- [x] Warehouse + fact/dimension design
- [x] Payroll concentration metrics
- [x] Win model scaffold
- [x] Team clustering scaffold
- [x] Dashboard starter
- [x] Add player-level WAR source (Baseball-Reference rWAR + Lahman approx fallback)
- [x] Build cost-per-WAR outputs

## Phase 2: Roster intelligence
- [ ] Add batting / pitching / fielding rollups at team-season level
- [ ] Build player value mart
- [ ] Add surplus-value and dead-money flags
- [ ] Add roster age and experience curves

## Phase 3: Game-level analytics
- [ ] Add Retrosheet or Statcast game data
- [ ] Build rolling 30-game form
- [ ] Analyze close-game and bullpen leverage performance
- [ ] Add opponent-adjusted performance

## Phase 4: Productization
- [ ] Dockerize
- [ ] Add CI/CD
- [ ] Publish dashboard
- [x] Thin read API over `current/` (`services/api`, #106)
- [x] Schedule refresh jobs
- [x] Shared artifact storage (`runs/` + `current/`) + dashboard remote load with local fallback
- [x] MLB Stats API majors ingest → versioned raw → warehouse (#108)
- [x] SportsDataIO Phase 0 ingest → aliases + `player_game_stat` spine (#128)
- [x] BenchOrStart waitlist + share-card shell (Phase 0)
- [x] Nightly ranked `fantasy/cards.jsonl` emitter (#111)

## Phase 5: Simulation
- [ ] Payroll redistribution simulator
- [ ] Monte Carlo season outcomes
- [ ] Contention-window classifier
