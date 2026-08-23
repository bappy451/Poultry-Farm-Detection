"""Run the validated pseudo-threshold calibration in CUDA half precision."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from ultralytics import YOLO
    from scripts import calibrate_pseudo_thresholds as base

    original_predict = YOLO.predict

    def predict_half(self, *args, **kwargs):
        device = str(kwargs.get("device", ""))
        if device.lower() == "cpu":
            raise ValueError("Half-precision calibration requires a CUDA device.")
        kwargs["half"] = True
        return original_predict(self, *args, **kwargs)

    YOLO.predict = predict_half  # type: ignore[method-assign]
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *(sys.argv[1:] if argv is None else argv)]
        return base.main()
    finally:
        sys.argv = old_argv
        YOLO.predict = original_predict  # type: ignore[method-assign]


if __name__ == "__main__":
    raise SystemExit(main())
