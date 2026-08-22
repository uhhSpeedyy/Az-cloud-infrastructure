from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from afl_ml.artifacts import save_json, save_model  # noqa: E402
from afl_ml.data_sources import (  # noqa: E402
    PublicDataClient,
    load_historical_matches,
    upcoming_round,
)
from afl_ml.database import ensure_schema, save_model_run, save_predictions  # noqa: E402
from afl_ml.features import (  # noqa: E402
    build_feature_frame,
    raw_same_match_correlations,
)
from afl_ml.modeling import train_model  # noqa: E402
from afl_ml.prediction import generate_predictions  # noqa: E402
from afl_ml.settings import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the AFL margin model")
    parser.add_argument("--start-season", type=int)
    parser.add_argument("--end-season", type=int)
    parser.add_argument("--holdout-season", type=int)
    parser.add_argument("--force", action="store_true", help="Refresh cached source files")
    parser.add_argument("--persist-db", action="store_true", help="Write the run to Azure SQL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    settings = replace(
        settings,
        start_season=args.start_season or settings.start_season,
        current_season=args.end_season or settings.current_season,
        holdout_season=args.holdout_season or settings.holdout_season,
    )
    client = PublicDataClient(settings, force=args.force)
    matches = load_historical_matches(
        client,
        settings.start_season,
        settings.current_season,
    )
    frame, engine = build_feature_frame(matches)
    bundle, report = train_model(frame, engine, settings.holdout_season)
    report["descriptive_same_match_correlations"] = raw_same_match_correlations(
        matches,
        settings.holdout_season,
    )
    manifest = client.source_manifest()
    manifest_digest = hashlib.sha256(
        "".join(item["sha256"] for item in manifest).encode("ascii")
    ).hexdigest()
    report["data_sources"] = {
        "wheelo": "https://www.wheeloratings.com/",
        "squiggle": "https://api.squiggle.com.au/",
        "source_files": len(manifest),
        "snapshot_sha256": manifest_digest,
        "historical_matches": len(matches),
        "latest_completed_match": max(match["start_time"] for match in matches).isoformat(),
    }

    games = upcoming_round(
        client.squiggle_games(settings.current_season, mutable=True)
    )
    predictions = generate_predictions(bundle, games)
    save_model(bundle, settings.model_path)
    save_json(report, settings.report_path)
    save_json(predictions, settings.predictions_path)

    database_status = "not requested"
    if args.persist_db:
        ensure_schema(settings)
        save_model_run(settings, report)
        save_predictions(settings, predictions)
        database_status = "saved"

    print(
        json.dumps(
            {
                "model_version": bundle.model_version,
                "historical_matches": len(matches),
                "training_matches": report["training_matches"],
                "holdout": report["holdout_metrics"],
                "upcoming_predictions": len(predictions["predictions"]),
                "database": database_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
