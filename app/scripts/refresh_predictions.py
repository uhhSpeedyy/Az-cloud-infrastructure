from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from afl_ml.service import refresh_prediction_snapshot  # noqa: E402
from afl_ml.settings import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the next AFL round predictions")
    parser.add_argument("--persist-db", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def refresh(*, persist_db: bool = False, force: bool = False) -> dict:
    settings = Settings()
    return refresh_prediction_snapshot(
        settings,
        persist_db=persist_db,
        force=force,
    )


def main() -> int:
    args = parse_args()
    payload = refresh(persist_db=args.persist_db, force=args.force)
    print(
        json.dumps(
            {
                "model_version": payload["model_version"],
                "round_name": payload["round_name"],
                "predictions": len(payload["predictions"]),
                "state_updates_applied": payload["state_updates_applied"],
                "database": "saved" if args.persist_db else "not requested",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
