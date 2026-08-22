from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # App Service settings are already provided as environment variables.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


APP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(APP_ROOT / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_root: Path = APP_ROOT
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AFL_DATA_DIR", APP_ROOT / "data"))
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AFL_ARTIFACTS_DIR", APP_ROOT / "artifacts"))
    )
    holdout_season: int = field(
        default_factory=lambda: int(os.getenv("AFL_HOLDOUT_SEASON", "2022"))
    )
    start_season: int = field(
        default_factory=lambda: int(os.getenv("AFL_START_SEASON", "2012"))
    )
    current_season: int = field(
        default_factory=lambda: int(os.getenv("AFL_CURRENT_SEASON", "2026"))
    )
    squiggle_contact: str = field(
        default_factory=lambda: os.getenv(
            "SQUIGGLE_CONTACT",
            "github.com/uhhSpeedyy/Az-cloud-infrastructure",
        )
    )
    database_enabled: bool = field(
        default_factory=lambda: _as_bool(os.getenv("AFL_DATABASE_ENABLED"), True)
    )
    database_read_enabled: bool = field(
        default_factory=lambda: _as_bool(
            os.getenv("AFL_DATABASE_READ_ENABLED"), False
        )
    )
    db_server: str | None = field(default_factory=lambda: os.getenv("DB_SERVER"))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "DB_one"))
    refresh_token: str | None = field(
        default_factory=lambda: os.getenv("AFL_REFRESH_TOKEN")
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "afl_margin_model.joblib"

    @property
    def predictions_path(self) -> Path:
        return self.artifacts_dir / "predictions.json"

    @property
    def report_path(self) -> Path:
        return self.artifacts_dir / "model_report.json"

    @property
    def user_agent(self) -> str:
        return (
            "AFL-ML-Prediction-Model/1.0 "
            f"(sam-speed.azurewebsites.net; contact: {self.squiggle_contact})"
        )

    def ensure_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
