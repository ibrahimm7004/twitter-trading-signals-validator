from __future__ import annotations

import math

from c4.utils.price_mapper import make_price_mapper


def test_price_mapper_linear_midpoint():
    ticks = [
        {"y_px": 0, "value": 100},
        {"y_px": 100, "value": 50},
    ]
    mapper = make_price_mapper(ticks, scale="linear")
    assert mapper(50) == 75.0


def test_price_mapper_log_midpoint():
    ticks = [
        {"y_px": 0, "value": 1000},
        {"y_px": 100, "value": 100},
    ]
    mapper = make_price_mapper(ticks, scale="log")
    assert math.isclose(mapper(50), 316.227766, rel_tol=1e-6)


def test_price_mapper_out_of_range_clamps_to_edge_segments():
    ticks = [
        {"y_px": 0, "value": 100},
        {"y_px": 100, "value": 50},
    ]
    mapper = make_price_mapper(ticks, scale="linear")
    assert mapper(-50) == 125.0
    assert mapper(150) == 25.0

