"""Unit tests for the model functions in utils.py and main.py."""

import numpy as np
import pytest
from shapely import Point, Polygon

from main import generate_seeds
from utils import (
    calculate_probability,
    exit_capacity_per_update,
    select_exiting_agents,
    collapse_hazard,
    collapse_probability,
    crowding_factor,
    exposure_factor,
    compute_max_risk,
    exposure_risk,
    get_nearest_exit_id,
    shooter_positions,
)

LINE = ((12.0, 11.0), (38.0, 90.0))


class ConstantRng:
    """Stand-in for numpy Generator that removes the noise term."""

    def uniform(self, low, high):
        return 1.0


def test_collapse_probability_risk_model_is_symmetric():
    p, g = 0.5, 0.8
    # alpha=1: dense is safer, isolated is more exposed
    assert collapse_probability(p, 1, g, 1.0) == pytest.approx(0.5 * (1 - g))
    assert collapse_probability(p, 0, g, 1.0) == pytest.approx(0.5 * (1 + g))
    # alpha=0: roles swap
    assert collapse_probability(p, 1, g, 0.0) == pytest.approx(0.5 * (1 + g))
    assert collapse_probability(p, 0, g, 0.0) == pytest.approx(0.5 * (1 - g))
    # alpha=0.5 or s=0.5: crowding has no effect
    assert collapse_probability(p, 1, g, 0.5) == pytest.approx(0.5)
    assert collapse_probability(p, 0.5, g, 0.0) == pytest.approx(0.5)


def test_collapse_probability_risk_model_is_clamped():
    assert collapse_probability(0.1, 0, 0.8, 1.0) == 1.0
    assert 0.0 <= collapse_probability(0.99, 1, 0.8, 1.0) <= 1.0


def test_collapse_probability_survival_model_matches_submitted_form():
    p, g = 0.5, 0.8
    assert collapse_probability(p, 0, g, 1.0, "survival") == pytest.approx(0.5)
    assert collapse_probability(p, 1, g, 1.0, "survival") == pytest.approx(0.1)
    assert collapse_probability(p, 1, g, 0.0, "survival") == pytest.approx(0.5)
    assert collapse_probability(0.9, 1, g, 1.0, "survival") == 0.0  # capped survival


def test_exposure_factor_is_one_on_the_line_and_decays():
    shooters = shooter_positions(LINE, 50)
    x, y = shooters[25]
    assert exposure_factor(Point(x, y), LINE, 30, 50) == pytest.approx(1.0, abs=1e-3)
    near = exposure_factor(Point(45, 50), LINE, 30, 50)
    far = exposure_factor(Point(180, 60), LINE, 30, 50)
    assert 1 > near > far > 0


def test_crowding_factor_limits():
    g = 0.8
    assert crowding_factor(1, g, 1.0) == pytest.approx(1 - g)
    assert crowding_factor(0, g, 1.0) == pytest.approx(1 + g)
    assert crowding_factor(1, g, 0.0) == pytest.approx(1 + g)
    assert crowding_factor(0.5, g, 0.0) == pytest.approx(1.0)
    assert crowding_factor(1, g, 0.5) == pytest.approx(1.0)


def test_collapse_hazard_on_the_line_equals_baseline():
    shooters = shooter_positions(LINE, 50)
    x, y = shooters[25]
    kw = dict(lambda_growth=0.0, time_scale=600, firing_line=LINE, sigma=30, gamma=0.8,
              alpha=0.5, tau_line=60, update_time=10)
    assert collapse_hazard(Point(x, y), 0, 0.5, **kw) == pytest.approx(10 / 60, abs=1e-3)
    kw["lambda_growth"] = 0.5
    assert collapse_hazard(Point(x, y), 600, 0.5, **kw) == pytest.approx(1.5 * 10 / 60, abs=1e-3)
    kw["tau_line"] = 1.0
    assert collapse_hazard(Point(x, y), 600, 0.5, **kw) == 1.0


def test_exit_capacity_matches_flow_times_width_times_dt():
    assert exit_capacity_per_update(1.3, 1.5, 10) == pytest.approx(19.5)


def test_select_exiting_agents_closest_first_with_carry_over():
    candidates = [(5.0, "c"), (1.0, "a"), (3.0, "b"), (9.0, "d")]
    chosen, credit = select_exiting_agents(candidates, credit=0.0, capacity=2.5)
    assert chosen == ["a", "b"]
    assert credit == pytest.approx(0.5)
    chosen, credit = select_exiting_agents(candidates, credit=credit, capacity=2.5)
    assert chosen == ["a", "b", "c"]  # 0.5 carried over makes 3
    assert credit == pytest.approx(0.0)


def test_select_exiting_agents_does_not_bank_a_burst():
    _, credit = select_exiting_agents([], credit=0.0, capacity=2.5)
    _, credit = select_exiting_agents([], credit=credit, capacity=2.5)
    assert credit <= 2.5
    chosen, _ = select_exiting_agents([(i, i) for i in range(20)], credit=credit, capacity=2.5)
    assert len(chosen) <= 5


def test_collapse_probability_rejects_unknown_model():
    with pytest.raises(ValueError):
        collapse_probability(0.5, 0.5, 0.8, 0.5, "foo")


def test_shooter_positions_span_the_segment():
    s = shooter_positions(LINE, 50)
    assert s.shape == (50, 2)
    assert tuple(s[0]) == LINE[0] and tuple(s[-1]) == LINE[1]
    spacing = np.linalg.norm(np.diff(s, axis=0), axis=1)
    assert np.allclose(spacing, spacing[0])


def test_compute_max_risk_is_the_maximum_along_the_line():
    sigma, n = 30, 50
    shooters = shooter_positions(LINE, n)
    along = [exposure_risk(x, y, shooters, sigma) for x, y in shooters]
    assert compute_max_risk(LINE, sigma, n) == pytest.approx(max(along))


def test_survival_reaches_p_min_on_the_firing_line():
    shooters = shooter_positions(LINE, 50)
    x, y = shooters[len(shooters) // 2]
    p = calculate_probability(
        Point(x, y), 0, 0.3, 600, LINE, rng=ConstantRng(), sigma=30,
    )
    assert p == pytest.approx(0.05)


def test_survival_increases_with_distance_and_decays_with_time():
    kw = dict(lambda_decay=0.3, time_scale=600, firing_line=LINE, rng=ConstantRng(), sigma=30)
    near = calculate_probability(Point(40, 50), 0, **kw)
    far = calculate_probability(Point(150, 50), 0, **kw)
    later = calculate_probability(Point(150, 50), 300, **kw)
    assert near < far
    assert later == pytest.approx(far * np.exp(-0.3 * 0.5))


def test_nearest_exit_is_chosen_deterministically_with_strong_determinism():
    exits = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(100, 0), (101, 0), (101, 1), (100, 1)])]
    rng = np.random.default_rng(0)
    journey, exit_id, dist = get_nearest_exit_id(
        Point(2, 0.5), exits, exit_ids=[10, 20], journey_ids=[1, 2], rng=rng, determinism_strength=50
    )
    assert (journey, exit_id) == (1, 10)
    assert dist == pytest.approx(1.0)


def test_generate_seeds_is_reproducible_and_distinct():
    a = generate_seeds(1234, 5)
    assert a == generate_seeds(1234, 5)
    assert len(set(a)) == 5
    assert all(0 <= s < 2**32 for s in a)
    assert a != generate_seeds(4321, 5)
