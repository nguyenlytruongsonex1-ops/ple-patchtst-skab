"""Print the Bike seasonal-naive sanity check. Not a training run."""

import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from quantile_pilot.data_bike import build_bike_dataset, load_hour_frame, seasonal_naive


def main() -> int:
    path = _EXP / "data" / "bike" / "hour.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run download_data.py")
    dataset = build_bike_dataset(load_hour_frame(path))
    payload = {
        "kind": "seasonal_naive_not_a_trained_model",
        "primary_model_metric": "MAE",
        "references": [seasonal_naive(dataset, 24), seasonal_naive(dataset, 168)],
        "windows": dataset["report"]["splits"],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
