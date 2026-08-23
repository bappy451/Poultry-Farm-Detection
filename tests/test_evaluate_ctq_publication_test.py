from scripts.evaluate_ctq_publication_test import METHODS, aggregate, paired_deltas


def test_aggregate_and_paired_deltas() -> None:
    rows = []
    for method, values in {
        "baseline": (0.70, 0.60),
        "control": (0.71, 0.61),
        "alpha2_weight0025": (0.73, 0.63),
    }.items():
        for seed, score in zip((7, 17), values):
            rows.append({
                "method": method, "seed": seed, "metrics/precision(B)": 0.8,
                "metrics/recall(B)": 0.8, "metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": score,
            })
    result = {item["method"]: item for item in aggregate(rows)}
    assert result["alpha2_weight0025"]["n_seeds"] == 2
    deltas = {item["comparison"]: item for item in paired_deltas(rows)}
    assert deltas["alpha2_weight0025_minus_baseline"]["mAP50-95_delta_mean"] == 0.03
    assert METHODS == ("baseline", "control", "alpha2_weight0025")
