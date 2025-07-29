import numpy as np
from signal_pipeline.vol_container_score import calculate_iv_rank, calculate_score


def test_calculate_iv_rank_basic():
    history = [10, 20, 30, 40]
    assert calculate_iv_rank(25, history) == 0.5
    assert calculate_iv_rank(5, history) == 0.0
    assert calculate_iv_rank(50, history) == 1.0


def test_calculate_score_weights():
    result = calculate_score(0.5, 0.2, 0.3, 0.1, 0.2)
    expected = (
        0.25 * 0.5 +
        0.20 * 0.2 +
        0.20 * 0.3 +
        0.15 * 0.1 +
        0.20 * 0.2
    )
    assert np.isclose(result, expected)
