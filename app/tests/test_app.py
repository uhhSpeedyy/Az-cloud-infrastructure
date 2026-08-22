from __future__ import annotations

from dataclasses import replace

from afl_ml.artifacts import save_json
from afl_ml.settings import Settings
from app import create_app


def test_landing_afl_and_health_render_snapshot(tmp_path):
    settings = replace(
        Settings(),
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        database_enabled=False,
    )
    settings.ensure_directories()
    save_json(
        {
            "model_version": "test-model",
            "generated_at": "2026-08-18T00:00:00+00:00",
            "season": 2026,
            "round_name": "Round 24",
            "predictions": [],
        },
        settings.predictions_path,
    )
    save_json(
        {
            "holdout_metrics": {
                "tip_accuracy": 0.6,
                "margin_mae": 22.0,
                "matches": 207,
            },
            "top_predictive_features": [
                {
                    "feature": "elo_diff",
                    "label": "Team strength rating",
                    "mae_increase_when_shuffled": 1.16,
                }
            ],
        },
        settings.report_path,
    )
    app = create_app(settings)
    client = app.test_client()

    landing = client.get("/")
    assert landing.status_code == 200
    assert b'href="/afl"' in landing.data
    assert b'href="/books"' in landing.data

    afl = client.get("/afl")
    assert afl.status_code == 200
    assert b"AFL Prediction Model" in afl.data
    assert b'class="home-link" href="/"' in afl.data
    assert b"Home</span>" in afl.data
    assert b"AFL ML" not in afl.data
    assert b"AFLML" not in afl.data
    assert b"By Sam Speed" in afl.data
    assert b"Data-led weekly forecasts" not in afl.data
    assert b"pre-match football indicators driving every call" not in afl.data
    assert b"Key prediction indicators" in afl.data
    assert b"extra average error after shuffling" in afl.data

    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["model_version"] == "test-model"
