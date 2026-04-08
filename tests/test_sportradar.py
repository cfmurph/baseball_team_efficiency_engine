"""
Tests for the Sportradar client and extractor parsing helpers.

All HTTP calls are mocked — no real API key required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.baseball_analytics.sportradar import SportradarClient, SportradarError
from pipeline.extract.pull_sportradar import (
    _parse_team_map,
    _parse_player_season,
    _parse_transactions,
    _parse_injuries,
)


# ---------------------------------------------------------------------------
# Fixtures — sample API response shapes
# ---------------------------------------------------------------------------

HIERARCHY_FIXTURE = {
    "league": {"alias": "MLB", "name": "Major League Baseball", "id": "mlb-id"},
    "leagues": [
        {
            "alias": "AL",
            "name": "American League",
            "id": "al-id",
            "divisions": [
                {
                    "alias": "E",
                    "name": "East",
                    "id": "ale-id",
                    "teams": [
                        {"id": "nyr-guid", "abbr": "NYY", "market": "New York", "name": "Yankees"},
                        {"id": "bos-guid", "abbr": "BOS", "market": "Boston", "name": "Red Sox"},
                    ],
                }
            ],
        }
    ],
}

SEASONAL_STATS_FIXTURE = {
    "id": "nyr-guid",
    "market": "New York",
    "name": "Yankees",
    "abbr": "NYY",
    "season": {"year": 2024, "type": "REG"},
    "players": [
            {
                "id": "player-a",
                "full_name": "Aaron Judge",
                "first_name": "Aaron",
                "last_name": "Judge",
                "position": "OF",
                "primary_position": "RF",
                "jersey_number": 99,
                "statistics": {
                    "hitting": {
                        "overall": {
                            "ap": 620,
                            "ab": 544,
                            "avg": 0.311,
                            "obp": 0.425,
                            "slg": 0.686,
                            "ops": 1.111,
                            "woba": 0.467,
                            "wraa": 52.1,
                            "wrc": 115.0,
                            "wrc_plus": 211.0,
                            "war": 11.4,
                            "bwar": 10.2,
                            "brwar": 0.5,
                            "fwar": 0.7,
                            "onbase": {"h": 169, "d": 28, "t": 1, "hr": 62, "bb": 78, "ibb": 9, "hbp": 3},
                            "outs": {"ktotal": 175},
                            "steal": {"stolen": 16},
                        }
                    },
                    "pitching": {"overall": {}},
                },
            },
            {
                "id": "player-b",
                "full_name": "Gerrit Cole",
                "first_name": "Gerrit",
                "last_name": "Cole",
                "position": "P",
                "primary_position": "SP",
                "jersey_number": 45,
                "statistics": {
                    "hitting": {"overall": {}},
                    "pitching": {
                        "overall": {
                            "ip_2": 200.0,
                            "era": "2.63",
                            "era_minus": 62.5,
                            "fip": 2.81,
                            "whip": 1.01,
                            "k9": 11.5,
                            "bb9": 2.1,
                            "kbb": 5.5,
                            "war": 7.2,
                            "onbase": {"hr9": 0.7},
                        }
                    },
                },
            },
        ],
}

TX_FIXTURE = {
    "players": [
        {
            "id": "player-a",
            "first_name": "Jordan",
            "last_name": "Montgomery",
            "position": "P",
            "primary_position": "SP",
            "status": "A",
            "transactions": [
                {
                    "id": "tx-001",
                    "desc": "NYY traded LHP Jordan Montgomery to STL.",
                    "effective_date": "2022-08-02",
                    "last_modified": "2022-08-02T21:56:06+00:00",
                    "transaction_type": "Traded",
                    "transaction_code": "TRD",
                    "from_team": {"name": "Yankees", "market": "New York", "abbr": "NYY", "id": "nyr-guid"},
                    "to_team": {"name": "Cardinals", "market": "St. Louis", "abbr": "STL", "id": "stl-guid"},
                }
            ],
        }
    ]
}

INJURY_FIXTURE = {
    "teams": [
        {
            "id": "nyr-guid",
            "abbr": "NYY",
            "players": [
                {
                    "id": "player-c",
                    "full_name": "Luis Severino",
                    "first_name": "Luis",
                    "last_name": "Severino",
                    "injury": {
                        "desc": "Right elbow inflammation",
                        "status": "D15",
                        "start_date": "2024-04-01",
                        "end_date": None,
                    },
                }
            ],
        }
    ]
}


# ---------------------------------------------------------------------------
# SportradarClient tests (mocked HTTP)
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    """A SportradarClient with a fake API key and response cache in tmp_path."""
    return SportradarClient(api_key="fake-key-123", access_level="trial", cache_dir=tmp_path)


def _mock_response(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


def test_client_requires_api_key(tmp_path):
    with pytest.raises(EnvironmentError, match="SPORTRADAR_API_KEY"):
        with patch.dict("os.environ", {}, clear=True):
            # Remove key from env if present
            import os
            os.environ.pop("SPORTRADAR_API_KEY", None)
            SportradarClient(api_key=None)


def test_client_uses_env_key(tmp_path):
    with patch.dict("os.environ", {"SPORTRADAR_API_KEY": "env-key"}):
        c = SportradarClient(cache_dir=tmp_path)
        assert c._key == "env-key"


def test_league_hierarchy(client):
    with patch.object(client._session, "get", return_value=_mock_response(HIERARCHY_FIXTURE)):
        result = client.league_hierarchy()
    assert result["league"]["alias"] == "MLB"
    assert len(result["leagues"]) == 1


def test_response_is_cached(client, tmp_path):
    """Second call should read from cache, not hit the network."""
    with patch.object(client._session, "get", return_value=_mock_response(HIERARCHY_FIXTURE)) as mock_get:
        client.league_hierarchy()
        client.league_hierarchy()
    assert mock_get.call_count == 1   # second call served from cache


def test_raises_on_404(client):
    with patch.object(client._session, "get", return_value=_mock_response({}, 404)):
        with pytest.raises(SportradarError) as exc_info:
            client.league_hierarchy()
    assert exc_info.value.status_code == 404


def test_seasonal_stats(client):
    with patch.object(client._session, "get", return_value=_mock_response(SEASONAL_STATS_FIXTURE)):
        result = client.seasonal_stats("nyr-guid", 2024)
    assert "players" in result
    assert len(result["players"]) == 2


def test_transactions(client):
    with patch.object(client._session, "get", return_value=_mock_response(TX_FIXTURE)):
        result = client.transactions(2022, 8, 2)
    assert "players" in result


# ---------------------------------------------------------------------------
# Parsing helper tests
# ---------------------------------------------------------------------------

def test_parse_team_map():
    df = _parse_team_map(HIERARCHY_FIXTURE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) >= {"sr_team_id", "sr_abbr", "sr_market", "sr_name"}
    assert "NYY" in df["sr_abbr"].values


def test_parse_player_season_hitter():
    df = _parse_player_season(SEASONAL_STATS_FIXTURE, "nyr-guid", 2024)
    assert len(df) == 2
    judge = df[df["full_name"] == "Aaron Judge"].iloc[0]
    assert float(judge["war"]) == pytest.approx(11.4)
    assert float(judge["wrc_plus"]) == pytest.approx(211.0)
    assert float(judge["hr"]) == 62


def test_parse_player_season_pitcher():
    df = _parse_player_season(SEASONAL_STATS_FIXTURE, "nyr-guid", 2024)
    cole = df[df["full_name"] == "Gerrit Cole"].iloc[0]
    assert float(cole["fip"]) == pytest.approx(2.81)
    assert float(cole["p_war"]) == pytest.approx(7.2)
    assert float(cole["k9"]) == pytest.approx(11.5)


def test_parse_player_season_team_id():
    df = _parse_player_season(SEASONAL_STATS_FIXTURE, "nyr-guid", 2024)
    assert (df["sr_team_id"] == "nyr-guid").all()
    assert (df["season_year"] == 2024).all()


def test_parse_transactions():
    df = _parse_transactions(TX_FIXTURE)
    assert len(df) == 1
    assert df.iloc[0]["transaction_code"] == "TRD"
    assert df.iloc[0]["from_team_abbr"] == "NYY"
    assert df.iloc[0]["to_team_abbr"] == "STL"


def test_parse_transactions_empty():
    df = _parse_transactions({})
    assert df.empty


def test_parse_injuries():
    df = _parse_injuries(INJURY_FIXTURE, "2024-04-10T12:00:00")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["team_abbr"] == "NYY"
    assert row["injury_status"] == "D15"
    assert "elbow" in row["injury_desc"].lower()


def test_parse_injuries_empty():
    df = _parse_injuries({"teams": []}, "2024-01-01")
    assert df.empty


# ---------------------------------------------------------------------------
# Crosswalk CSV sanity
# ---------------------------------------------------------------------------

def test_crosswalk_csv_exists():
    path = Path("data/crosswalks/sportradar_team_map.csv")
    assert path.exists(), "data/crosswalks/sportradar_team_map.csv not found"


def test_crosswalk_has_all_30_teams():
    path = Path("data/crosswalks/sportradar_team_map.csv")
    if not path.exists():
        pytest.skip("Crosswalk not present")
    df = pd.read_csv(path)
    assert len(df) >= 30, f"Expected >= 30 teams, got {len(df)}"
    assert "sr_team_id" in df.columns
    assert "lahman_team_id" in df.columns
