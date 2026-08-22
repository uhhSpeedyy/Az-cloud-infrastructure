from __future__ import annotations

import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from afl_ml.database import database_connection  # noqa: E402
from afl_ml.settings import Settings  # noqa: E402


if __name__ == "__main__":
    statement = (APP_ROOT / "sql" / "002_grant_app_identity.sql").read_text(
        encoding="utf-8"
    )
    with database_connection(Settings()) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
    print("Sam-Speed managed identity has least-privilege AFL table access.")
