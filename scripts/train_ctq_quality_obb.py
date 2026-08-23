"""Run quality-aware CTQ pseudo-OBB adaptation without editing existing trainers."""

from __future__ import annotations

import argparse
import sys


def extract_quality_alpha(argv: list[str]) -> tuple[float, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--quality-alpha", type=float, default=1.0)
    parsed, remaining = parser.parse_known_args(argv)
    if parsed.quality_alpha < 0:
        raise ValueError("--quality-alpha must be non-negative")
    return float(parsed.quality_alpha), remaining


def main(argv: list[str] | None = None) -> int:
    alpha, remaining = extract_quality_alpha(list(sys.argv[1:] if argv is None else argv))
    from scripts import train_ema_pseudo_obb as base
    from scripts.ctq_obb_model_safe import SafeCTQOBBTrainer

    original_get_model = SafeCTQOBBTrainer.get_model

    def get_model_with_alpha(self, *args, **kwargs):
        self._adapt_options["quality_alpha"] = alpha
        return original_get_model(self, *args, **kwargs)

    SafeCTQOBBTrainer.get_model = get_model_with_alpha  # type: ignore[method-assign]
    base.PseudoLabelEMADomainAdaptOBBTrainer = SafeCTQOBBTrainer
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        return base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
