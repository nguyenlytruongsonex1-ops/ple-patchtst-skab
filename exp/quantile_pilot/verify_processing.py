import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from quantile_pilot.processing import main

if __name__ == "__main__":
    raise SystemExit(main())
