from __future__ import annotations

from scripts.calibrate_pseudo_thresholds import choose_threshold, match_predictions, score_threshold


def test_threshold_selection_prefers_recall_at_required_precision():
    records = [(0.95, True), (0.90, False), (0.80, True), (0.70, True)]
    threshold, score, fallback = choose_threshold(
        records,
        target_count=3,
        precision_target=0.90,
        min_predictions=1,
        candidates=[0.70, 0.80, 0.90, 0.95],
        fallback=0.90,
    )
    assert threshold == 0.95
    assert score["precision"] == 1.0
    assert score["recall"] == 1 / 3
    assert fallback is False


def test_threshold_falls_back_when_no_candidate_has_support():
    threshold, score, fallback = choose_threshold(
        [(0.95, True)],
        target_count=1,
        precision_target=0.90,
        min_predictions=10,
        candidates=[0.50, 0.90],
        fallback=0.90,
    )
    assert threshold == 0.90
    assert score["fallback_threshold"] == 0.90
    assert fallback is True


def test_obb_matching_is_class_specific_and_confidence_ordered():
    target = [(0, __import__("numpy").array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float))]
    predictions = [
        (1, 0.99, target[0][1]),
        (0, 0.80, target[0][1]),
    ]
    records = match_predictions(predictions, target, iou_threshold=0.5)
    assert records[1][0][1] is False
    assert records[0][0][1] is True


def test_score_threshold_counts_only_selected_predictions():
    score = score_threshold([(0.95, True), (0.60, False)], target_count=1, threshold=0.90)
    assert score["predictions"] == 1
    assert score["true_positives"] == 1
    assert score["precision"] == 1.0
