"""Unit tests for the model functions in utils.py and main.py."""

import numpy as np
import pytest
from shapely import Point, Polygon

from main import generate_seeds
from utils import (
    adjusted_probability,
    calculate_probability,
    compute_max_risk,
    get_nearest_exit_id,
)


class ConstantRng:
    """Stand-in for numpy Generator that removes the noise term."""

    def uniform(self, low, high):
        return 1.0


@pytest.fixture
def area():
    return Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])


def test_adjusted_probability_shielding_and_targeting():
    base = 0.5
    assert adjusted_probability(base, shielding=0, gamma=0.8, alpha=1.0) == base
    assert adjusted_probability(base, shielding=1, gamma=0.8, alpha=1.0) == pytest.approx(0.9)
    # alpha=0: dense clusters are more exposed, isolated agents are safer
    assert adjusted_probability(base, shielding=1, gamma=0.8, alpha=0.0) == base
    assert adjusted_probability(base, shielding=0, gamma=0.8, alpha=0.0) == pytest.approx(0.9)


def test_adjusted_probability_is_clamped():
    assert adjusted_probability(0.9, shielding=1, gamma=0.8, alpha=1.0) == 1.0


def test_compute_max_risk_matches_direct_sum():
    sigma, n = 30, 50
    ymin, ymax = -7.66, 133.381
    ys = np.linspace(ymin, ymax, n)
    yc = 0.5 * (ymin + ymax)
    expected = sum(1 / (1 + (yc - y) ** 2 / sigma**2) for y in ys)
    assert compute_max_risk(0, ymin, ymax, sigma, n) == pytest.approx(expected)


def test_survival_reaches_p_min_at_shooting_line(area):
    _, ymin, _, ymax = area.bounds
    p = calculate_probability(
        Point(0, 0.5 * (ymin + ymax)), 0, 0.3, 600, area,
        shielding=0, gamma=0, alpha=0, rng=ConstantRng(), sigma=30,
    )
    assert p == pytest.approx(0.05)


def test_survival_increases_with_distance_and_decays_with_time(area):
    kw = dict(lambda_decay=0.3, time_scale=600, walkable_area=area,
              shielding=0, gamma=0, alpha=0, rng=ConstantRng(), sigma=30)
    near = calculate_probability(Point(5, 50), 0, **kw)
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
