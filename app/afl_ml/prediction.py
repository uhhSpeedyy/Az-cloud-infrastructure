from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .data_sources import canonical_team
from .modeling import ModelBundle


def fixture_feature_frame(
    bundle: ModelBundle,
    games: list[dict[str, Any]],
) -> pd.DataFrame:
    engine = bundle.feature_engine.copy()
    rows: list[dict[str, Any]] = []
    for game in games:
        fixture = {
            "season": int(game.get("year") or game["start_time"].year),
            "round_number": int(game.get("round") or 0),
            "round_name": game.get("roundname") or f"Round {game.get('round', '')}",
            "start_time": game["start_time"],
            "venue": game.get("venue") or "Venue TBC",
            "home_team": canonical_team(game.get("hteam")),
            "away_team": canonical_team(game.get("ateam")),
            "is_final": bool(game.get("is_final")),
        }
        rows.append(
            {
                "source_game_id": str(game.get("id")),
                **fixture,
                **engine.features_for_fixture(fixture),
            }
        )
    return pd.DataFrame(rows)


def generate_predictions(
    bundle: ModelBundle,
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    if not games:
        return {
            "model_version": bundle.model_version,
            "generated_at": generated_at,
            "round_name": "No upcoming round",
            "predictions": [],
        }

    frame = fixture_feature_frame(bundle, games)
    model_output = bundle.predict(frame)
    predictions: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        margin = float(model_output["margin"][index])
        home_probability = round(float(model_output["home_probability"][index]) * 100.0, 1)
        draw_probability = round(float(model_output["draw_probability"][index]) * 100.0, 1)
        # Preserve a displayed three-way total of exactly 100% after rounding.
        away_probability = round(100.0 - home_probability - draw_probability, 1)
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        predicted_winner = home_team if margin >= 0 else away_team
        predictions.append(
            {
                "game_id": str(row["source_game_id"]),
                "season": int(row["season"]),
                "round_number": int(row["round_number"]),
                "round_name": str(row["round_name"]),
                "start_time": row["start_time"].isoformat(),
                "venue": str(row["venue"]),
                "home_team": home_team,
                "away_team": away_team,
                "home_win_probability": home_probability,
                "away_win_probability": away_probability,
                "draw_probability": draw_probability,
                "expected_home_margin": round(margin, 1),
                "expected_margin": round(abs(margin), 1),
                "predicted_winner": predicted_winner,
                "interval_80_low": round(float(model_output["interval_low"][index]), 1),
                "interval_80_high": round(float(model_output["interval_high"][index]), 1),
                "factors": bundle.explain_row(row, limit=5),
                "model_version": bundle.model_version,
                "generated_at": generated_at,
            }
        )
    return {
        "model_version": bundle.model_version,
        "generated_at": generated_at,
        "round_name": str(frame.iloc[0]["round_name"]),
        "season": int(frame.iloc[0]["season"]),
        "predictions": predictions,
    }
