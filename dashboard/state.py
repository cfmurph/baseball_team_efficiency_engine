"""Shared ``st.session_state`` keys for front-office dashboard pages.

These names are the deep-link / cross-page contract. Widget ``key=`` values
on season, team, and league controls must use these strings so a selection
on Overview is still selected on Team Deep Dive (and the reverse).

Keys
----
season_year      int          Selected season (Overview, Deep Dive, Roster,
                              Contract Watch, What-If).
selected_team    str          Single franchise (Deep Dive, What-If; Roster
                              and Contract Watch default when not "All Teams").
selected_league  str          ``All`` | ``AL`` | ``NL`` (Overview).
nav_page         str          Active section label from NAV_PAGES.
"""
from __future__ import annotations

SEASON_YEAR = "season_year"
SELECTED_TEAM = "selected_team"
SELECTED_LEAGUE = "selected_league"
NAV_PAGE = "nav_page"

SHARED_STATE_KEYS = (SEASON_YEAR, SELECTED_TEAM, SELECTED_LEAGUE, NAV_PAGE)
