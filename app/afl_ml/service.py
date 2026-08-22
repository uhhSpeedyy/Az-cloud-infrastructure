from __future__ import annotations

from .artifacts import load_json, load_model, save_json
from .data_sources import PublicDataClient, load_historical_matches, upcoming_round
from .database import save_model_run, save_predictions
from .features import update_engine_with_matches
from .prediction import generate_predictions
from .settings import Settings


def refresh_prediction_snapshot(
    settings: Settings,
    *,
    persist_db: bool = False,
    force: bool = False,
) -> dict:
    bundle = load_model(settings.model_path)
    client = PublicDataClient(settings, force=force)
    current_matches = load_historical_matches(
        client,
        settings.current_season,
        settings.current_season,
    )
    updated_matches = update_engine_with_matches(bundle.feature_engine, current_matches)
    games = upcoming_round(
        client.squiggle_games(settings.current_season, mutable=True)
    )
    payload = generate_predictions(bundle, games)
    payload["state_updates_applied"] = updated_matches
    save_json(payload, settings.predictions_path)
    if persist_db:
        report = load_json(settings.report_path)
        save_model_run(settings, report)
        save_predictions(settings, payload)
    return payload
