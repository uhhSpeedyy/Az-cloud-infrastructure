from __future__ import annotations

import json
import os
import struct
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import pyodbc
from azure.identity import DefaultAzureCredential

from .settings import Settings


SQL_COPT_SS_ACCESS_TOKEN = 1256
SQL_SCOPE = "https://database.windows.net/.default"


def _connection_string(settings: Settings) -> str:
    return (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER=tcp:{settings.db_server},1433;"
        f"DATABASE={settings.db_name};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=5;"
    )


@contextmanager
def database_connection(settings: Settings) -> Iterator[pyodbc.Connection]:
    if not settings.database_enabled:
        raise RuntimeError("AFL database persistence is disabled")
    if not settings.db_server or not settings.db_name:
        raise RuntimeError("DB_SERVER and DB_NAME must be configured")

    connection_string = _connection_string(settings)
    if os.getenv("WEBSITE_INSTANCE_ID") or os.getenv("IDENTITY_ENDPOINT"):
        connection = pyodbc.connect(
            connection_string + "Authentication=ActiveDirectoryMsi;",
            autocommit=False,
        )
    else:
        token = DefaultAzureCredential().get_token(SQL_SCOPE).token
        token_bytes = token.encode("utf-16-le")
        token_struct = struct.pack(
            f"<I{len(token_bytes)}s",
            len(token_bytes),
            token_bytes,
        )
        connection = pyodbc.connect(
            connection_string,
            attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct},
            autocommit=False,
        )
    try:
        yield connection
    finally:
        connection.close()


SCHEMA_STATEMENTS = (
    """
    IF OBJECT_ID(N'dbo.AflModelRuns', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.AflModelRuns (
            model_version NVARCHAR(64) NOT NULL PRIMARY KEY,
            trained_at DATETIMEOFFSET NOT NULL,
            holdout_season SMALLINT NOT NULL,
            training_matches INT NOT NULL,
            holdout_tip_accuracy DECIMAL(7,6) NULL,
            holdout_margin_mae DECIMAL(8,3) NULL,
            report_json NVARCHAR(MAX) NOT NULL,
            created_at DATETIMEOFFSET NOT NULL DEFAULT SYSUTCDATETIME(),
            CONSTRAINT CK_AflModelRuns_ReportJson CHECK (ISJSON(report_json) = 1)
        );
    END
    """,
    """
    IF OBJECT_ID(N'dbo.AflPredictions', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.AflPredictions (
            game_id NVARCHAR(40) NOT NULL PRIMARY KEY,
            model_version NVARCHAR(64) NOT NULL,
            season SMALLINT NOT NULL,
            round_number SMALLINT NOT NULL,
            round_name NVARCHAR(80) NOT NULL,
            start_time DATETIMEOFFSET NOT NULL,
            venue NVARCHAR(120) NOT NULL,
            home_team NVARCHAR(80) NOT NULL,
            away_team NVARCHAR(80) NOT NULL,
            home_win_probability DECIMAL(6,3) NOT NULL,
            away_win_probability DECIMAL(6,3) NOT NULL,
            draw_probability DECIMAL(6,3) NOT NULL,
            expected_home_margin DECIMAL(8,3) NOT NULL,
            interval_80_low DECIMAL(8,3) NULL,
            interval_80_high DECIMAL(8,3) NULL,
            factors_json NVARCHAR(MAX) NOT NULL,
            generated_at DATETIMEOFFSET NOT NULL,
            updated_at DATETIMEOFFSET NOT NULL DEFAULT SYSUTCDATETIME(),
            CONSTRAINT FK_AflPredictions_ModelRun FOREIGN KEY (model_version)
                REFERENCES dbo.AflModelRuns(model_version),
            CONSTRAINT CK_AflPredictions_FactorsJson CHECK (ISJSON(factors_json) = 1),
            CONSTRAINT CK_AflPredictions_HomeProbability CHECK
                (home_win_probability >= 0 AND home_win_probability <= 100),
            CONSTRAINT CK_AflPredictions_AwayProbability CHECK
                (away_win_probability >= 0 AND away_win_probability <= 100),
            CONSTRAINT CK_AflPredictions_DrawProbability CHECK
                (draw_probability >= 0 AND draw_probability <= 100),
            CONSTRAINT CK_AflPredictions_ProbabilityTotal CHECK
                (ABS(home_win_probability + away_win_probability + draw_probability - 100) <= 0.101)
        );
        CREATE INDEX IX_AflPredictions_StartTime ON dbo.AflPredictions(start_time);
    END
    """,
    """
    IF OBJECT_ID(N'dbo.AflPredictionSnapshots', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.AflPredictionSnapshots (
            snapshot_id NVARCHAR(128) NOT NULL PRIMARY KEY,
            model_version NVARCHAR(64) NOT NULL,
            season SMALLINT NULL,
            round_name NVARCHAR(80) NOT NULL,
            prediction_count SMALLINT NOT NULL,
            generated_at DATETIMEOFFSET NOT NULL,
            payload_json NVARCHAR(MAX) NOT NULL,
            created_at DATETIMEOFFSET NOT NULL DEFAULT SYSUTCDATETIME(),
            CONSTRAINT FK_AflPredictionSnapshots_ModelRun FOREIGN KEY (model_version)
                REFERENCES dbo.AflModelRuns(model_version),
            CONSTRAINT CK_AflPredictionSnapshots_PayloadJson CHECK (ISJSON(payload_json) = 1)
        );
        CREATE INDEX IX_AflPredictionSnapshots_GeneratedAt
            ON dbo.AflPredictionSnapshots(generated_at DESC);
    END
    """,
)


def ensure_schema(settings: Settings) -> None:
    with database_connection(settings) as connection:
        cursor = connection.cursor()
        try:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()


def save_model_run(settings: Settings, report: dict[str, Any]) -> None:
    metrics = report["holdout_metrics"]
    statement = """
    MERGE dbo.AflModelRuns AS target
    USING (SELECT ? AS model_version) AS source
        ON target.model_version = source.model_version
    WHEN MATCHED THEN UPDATE SET
        trained_at = ?, holdout_season = ?, training_matches = ?,
        holdout_tip_accuracy = ?, holdout_margin_mae = ?, report_json = ?
    WHEN NOT MATCHED THEN INSERT (
        model_version, trained_at, holdout_season, training_matches,
        holdout_tip_accuracy, holdout_margin_mae, report_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    values = (
        report["model_version"],
        report["trained_at"],
        report["holdout_season"],
        report["training_matches"],
        metrics["tip_accuracy"],
        metrics["margin_mae"],
        json.dumps(report, allow_nan=False),
    )
    with database_connection(settings) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(statement, values + values)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()


def save_predictions(settings: Settings, payload: dict[str, Any]) -> None:
    statement = """
    MERGE dbo.AflPredictions AS target
    USING (SELECT ? AS game_id) AS source
        ON target.game_id = source.game_id
    WHEN MATCHED THEN UPDATE SET
        model_version = ?, season = ?, round_number = ?, round_name = ?,
        start_time = ?, venue = ?, home_team = ?, away_team = ?,
        home_win_probability = ?, away_win_probability = ?, draw_probability = ?,
        expected_home_margin = ?, interval_80_low = ?, interval_80_high = ?,
        factors_json = ?, generated_at = ?, updated_at = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN INSERT (
        game_id, model_version, season, round_number, round_name,
        start_time, venue, home_team, away_team,
        home_win_probability, away_win_probability, draw_probability,
        expected_home_margin, interval_80_low, interval_80_high,
        factors_json, generated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    snapshot_statement = """
    MERGE dbo.AflPredictionSnapshots AS target
    USING (SELECT ? AS snapshot_id) AS source
        ON target.snapshot_id = source.snapshot_id
    WHEN MATCHED THEN UPDATE SET
        model_version = ?, season = ?, round_name = ?, prediction_count = ?,
        generated_at = ?, payload_json = ?
    WHEN NOT MATCHED THEN INSERT (
        snapshot_id, model_version, season, round_name, prediction_count,
        generated_at, payload_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    with database_connection(settings) as connection:
        cursor = connection.cursor()
        try:
            for prediction in payload.get("predictions", []):
                values = (
                    prediction["game_id"],
                    prediction["model_version"],
                    prediction["season"],
                    prediction["round_number"],
                    prediction["round_name"],
                    prediction["start_time"],
                    prediction["venue"],
                    prediction["home_team"],
                    prediction["away_team"],
                    prediction["home_win_probability"],
                    prediction["away_win_probability"],
                    prediction["draw_probability"],
                    prediction["expected_home_margin"],
                    prediction["interval_80_low"],
                    prediction["interval_80_high"],
                    json.dumps(prediction["factors"], allow_nan=False),
                    prediction["generated_at"],
                )
                cursor.execute(statement, values + values)
            snapshot_id = f"{payload['model_version']}:{payload['generated_at']}"
            snapshot_values = (
                snapshot_id,
                payload["model_version"],
                payload.get("season"),
                payload.get("round_name", "No upcoming round"),
                len(payload.get("predictions", [])),
                payload["generated_at"],
                json.dumps(payload, allow_nan=False),
            )
            cursor.execute(snapshot_statement, snapshot_values + snapshot_values)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()


def load_predictions(settings: Settings) -> dict[str, Any] | None:
    statement = """
    SELECT TOP (1) payload_json
    FROM dbo.AflPredictionSnapshots
    ORDER BY generated_at DESC, created_at DESC;
    """
    with database_connection(settings) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(statement)
            row = cursor.fetchone()
        finally:
            cursor.close()
    if not row:
        return None
    return json.loads(row.payload_json)


def database_health(settings: Settings) -> tuple[bool, str]:
    try:
        with database_connection(settings) as connection:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            finally:
                cursor.close()
        return True, "connected"
    except Exception as exc:
        return False, str(exc)
