from __future__ import annotations

from datetime import datetime, timedelta, timezone

from afl_ml.features import build_feature_frame


def _match(match_id: str, home: str, away: str, day: int, round_number: int, margin: int):
    home_score = 80 + margin
    away_score = 80
    stats = {
        "Score": None,
        "xScore": 80.0,
        "Inside50s": 52.0,
        "TotalClearances": 38.0,
        "CentreClearances": 12.0,
        "ContestedPossessions": 130.0,
        "DisposalEfficiency": 72.0,
        "MetresGained": 5_000.0,
        "PressureActs": 270.0,
        "Turnovers": 65.0,
        "Intercepts": 62.0,
        "ContestedMarks": 12.0,
        "ShotsAtGoal": 24.0,
        "RatingPoints": 220.0,
        "Experience": 85.0,
        "GroundBallGets": 90.0,
        "PostClearanceContestedPossessions": 70.0,
    }
    return {
        "match_id": match_id,
        "source_game_id": match_id,
        "season": 2021,
        "round_number": round_number,
        "round_name": f"Round {round_number}",
        "start_time": datetime(2021, 3, 1, tzinfo=timezone.utc) + timedelta(days=day),
        "venue": "M.C.G.",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_stats": dict(stats),
        "away_stats": dict(stats),
        "home_players": [],
        "away_players": [],
        "is_final": False,
    }


def test_same_round_results_do_not_leak_between_feature_snapshots():
    matches = [
        _match("one", "Carlton", "Richmond", 0, 1, 20),
        _match("two", "Carlton", "Collingwood", 1, 1, -10),
        _match("three", "Carlton", "Essendon", 8, 2, 5),
    ]
    frame, _ = build_feature_frame(matches)
    assert frame.loc[0, "prior_games_min"] == 0
    assert frame.loc[1, "prior_games_min"] == 0
    assert frame.loc[2, "prior_games_min"] >= 0
    assert frame.loc[0, "form_margin_short_diff"] != frame.loc[0, "form_margin_short_diff"]
    assert frame.loc[1, "form_margin_short_diff"] != frame.loc[1, "form_margin_short_diff"]
