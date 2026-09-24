import math

import pytest

from jevbench import composite_v13 as score


def test_exact_item_chance_baselines():
    assert score.TIER_CHANCES == pytest.approx({
        "easy": 0.28402777777777755,
        "standard": 0.3166666666666668,
        "judge": 0.29223744292237613,
        "hard": 0.3364393939393943,
    })
    assert {tier: sum(counts.values()) for tier, counts in score.TIER_OPTION_COUNTS.items()} == {
        "easy": 72, "standard": 96, "judge": 146, "hard": 220,
    }


def test_chance_correction_and_clipping():
    assert score.chance_corrected_accuracy(0.5, 0.5) == 0
    assert score.chance_corrected_accuracy(0.75, 0.5) == 50
    assert score.chance_corrected_accuracy(1, 0.5) == 100


def test_tier_weights_are_unchanged():
    assert score.intelligence({"easy": 1, "standard": 1, "judge": 1, "hard": 0}) == pytest.approx(70)
    assert score.intelligence({"easy": 1, "standard": 1, "judge": 1, "hard": None}) == pytest.approx(100)


def test_near_chance_penalty_boundary_and_shape():
    axes = {"intelligence": 50, "calibration": 80, "speed": 80, "cost": 80}
    unpenalized = math.exp(sum(math.log(v) for v in axes.values()) / 4)
    assert score.jevbench_score(axes) == pytest.approx(unpenalized)
    axes["intelligence"] = 25
    base = math.exp(sum(math.log(v) for v in axes.values()) / 4)
    assert score.jevbench_score(axes) == pytest.approx(base * 0.25)


def test_unchanged_speed_cost_and_calibration():
    assert score.speed_point(0.1) == 100
    assert score.speed_point(1) == pytest.approx(80)
    assert score.cost(0.001) == 100
    assert score.cost(0.01) == pytest.approx(70)
    assert score.calibration(0.1, 0.2) == 80
