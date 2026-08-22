from __future__ import annotations

from datetime import datetime, timezone

from afl_ml.data_sources import canonical_team, columnar_records, upcoming_round


def test_columnar_records_handles_lists_and_scalars():
    rows = columnar_records({"team": ["A", "B"], "round": 3, "score": [80, 71]})
    assert rows == [
        {"team": "A", "round": 3, "score": 80},
        {"team": "B", "round": 3, "score": 71},
    ]


def test_team_aliases_are_canonical():
    assert canonical_team("Brisbane") == "Brisbane Lions"
    assert canonical_team("Greater Western Sydney") == "GWS"
    assert canonical_team("West Coast Eagles") == "West Coast"


def test_upcoming_round_only_returns_first_future_round():
    games = [
        {
            "id": 1,
            "year": 2026,
            "round": 24,
            "date": "2026-08-20 19:30:00",
            "tz": "+10:00",
            "complete": 0,
            "is_final": 0,
        },
        {
            "id": 2,
            "year": 2026,
            "round": 25,
            "date": "2026-08-28 19:30:00",
            "tz": "+10:00",
            "complete": 0,
            "is_final": 0,
        },
    ]
    result = upcoming_round(games, now=datetime(2026, 8, 18, tzinfo=timezone.utc))
    assert [game["id"] for game in result] == [1]
