#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


# When this file is executed directly, its own filename shadows the package name.
# Add the project src directory explicitly so `from poultry_geo...` resolves to the
# package rather than back to this wrapper module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path = [path for path in sys.path if path != str(SCRIPT_DIR)]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.modules.pop("poultry_geo", None)

from poultry_geo.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
