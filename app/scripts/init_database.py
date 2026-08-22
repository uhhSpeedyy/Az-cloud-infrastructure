from __future__ import annotations

import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from afl_ml.database import ensure_schema  # noqa: E402
from afl_ml.settings import Settings  # noqa: E402


if __name__ == "__main__":
    ensure_schema(Settings())
    print("AFL model tables are ready in Azure SQL.")
