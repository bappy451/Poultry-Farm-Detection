"""Run reliability-gated CTQ adaptation without modifying existing trainers."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--quality-alpha", type=float, default=1.0)
    parser.add_argument("--teacher-jitter-views", type=int, default=2)
    parser.add_argument("--teacher-jitter-iou", type=float, default=0.70)
    parser.add_argument("--teacher-jitter-agreement", type=float, default=1.0)
    parsed, remaining = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if parsed.quality_alpha < 0 or parsed.teacher_jitter_views < 0:
        raise ValueError("quality alpha and jitter views must be non-negative")
    if not 0.0 < parsed.teacher_jitter_iou <= 1.0 or not 0.0 < parsed.teacher_jitter_agreement <= 1.0:
        raise ValueError("teacher jitter IoU/agreement must be in (0, 1]")
    from scripts import train_ema_pseudo_obb as base
    from scripts.ctq_reliable_obb import ReliableCTQOBBTrainer

    original = ReliableCTQOBBTrainer.get_model

    def get_model_with_options(self, *args, **kwargs):
        self._adapt_options.update({
            "quality_alpha": parsed.quality_alpha,
            "teacher_jitter_views": parsed.teacher_jitter_views,
            "teacher_jitter_iou": parsed.teacher_jitter_iou,
            "teacher_jitter_agreement": parsed.teacher_jitter_agreement,
        })
        return original(self, *args, **kwargs)

    ReliableCTQOBBTrainer.get_model = get_model_with_options  # type: ignore[method-assign]
    base.PseudoLabelEMADomainAdaptOBBTrainer = ReliableCTQOBBTrainer
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        return base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
