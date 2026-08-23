"""Run the standalone reliability- and balance-controlled CTQ OBB trainer."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--quality-alpha", type=float, default=1.0)
    parser.add_argument("--teacher-jitter-views", type=int, default=2)
    parser.add_argument("--teacher-jitter-iou", type=float, default=0.70)
    parser.add_argument("--teacher-jitter-agreement", type=float, default=1.0)
    parser.add_argument("--pseudo-background-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-class-weight-min", type=float, default=0.5)
    parser.add_argument("--pseudo-class-weight-max", type=float, default=2.0)
    parsed, remaining = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if parsed.quality_alpha < 0 or parsed.teacher_jitter_views < 0:
        raise ValueError("quality alpha and teacher jitter views must be non-negative")
    if not 0 < parsed.teacher_jitter_iou <= 1 or not 0 < parsed.teacher_jitter_agreement <= 1 or not 0 < parsed.pseudo_background_weight <= 1:
        raise ValueError("jitter agreement/IoU and pseudo background weight must be in (0, 1]")
    if parsed.pseudo_class_weight_min <= 0 or parsed.pseudo_class_weight_max < parsed.pseudo_class_weight_min:
        raise ValueError("Invalid pseudo class-weight bounds")
    from scripts import train_ema_pseudo_obb as base
    from scripts.ctq_reliable_balanced_obb import BalancedReliableCTQOBBTrainer
    original = BalancedReliableCTQOBBTrainer.get_model
    def get_model_with_options(self, *args, **kwargs):
        self._adapt_options.update(vars(parsed))
        return original(self, *args, **kwargs)
    BalancedReliableCTQOBBTrainer.get_model = get_model_with_options  # type: ignore[method-assign]
    base.PseudoLabelEMADomainAdaptOBBTrainer = BalancedReliableCTQOBBTrainer
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        return base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
