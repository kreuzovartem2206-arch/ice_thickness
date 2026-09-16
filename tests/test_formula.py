import numpy as np

from src.daily_multisat_l2_sit import compute_sit


def test_compute_sit_matches_hydrostatic_formula():
    freeboard = np.array([0.20])
    snow_depth = np.array([0.18])
    snow_density = np.array([320.0])
    ice_density = np.array([915.0])

    expected = (1024.0 * freeboard + snow_density * snow_depth) / (1024.0 - ice_density)
    actual = compute_sit(freeboard, snow_depth, snow_density, ice_density)

    assert np.allclose(actual, expected)
