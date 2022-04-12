from pathlib import Path

import pytest

from irnl_rdt_correction.irnl_rdt_correction import main as irnl_correct
from tests.helpers import (
    generate_pseudo_model, generate_errortable, get_some_magnet_names, VALUE, STRENGTH, IP, \
    EPS, FIELD
)


@pytest.mark.parametrize('x', (2, 0))
@pytest.mark.parametrize('y', (1.5, 0))
def test_general_feeddown(tmp_path: Path, x: float, y: float):
    """Test feeddown functionality from decapoles to octupoles and sextupoles."""
    # Parameters -----------------------------------------------------------
    accel = 'lhc'

    correct_ips = (1, 3)
    error_value = 2
    n_magnets = 4
    n_ips = 4
    n_sides = 2

    # Setup ----------------------------------------------------------------
    optics = generate_pseudo_model(
        accel=accel, n_ips=n_ips, n_magnets=n_magnets, x=x, y=y)
    errors = generate_errortable(
        index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
    )
    errors["K4L"] = error_value  # normal decapole errors

    # Correction ---------------------------------------------------------------
    rdts = "f4000", "f3001"
    _, df_corrections = irnl_correct(
        accel=accel,
        optics=[optics],
        errors=[errors],
        beams=[1],
        rdts=rdts,
        output=tmp_path / "correct",
        feeddown=0,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    _, df_corrections_fd1 = irnl_correct(
        accel=accel,
        optics=[optics],
        errors=[errors],
        beams=[1],
        rdts=rdts,
        output=tmp_path / "correct_fd1",
        feeddown=1,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    errors["K4L"] = 0
    errors["K5L"] = error_value  # normal dodecapole errors
    _, df_corrections_fd2 = irnl_correct(
        accel=accel,
        optics=[optics],
        errors=[errors],
        beams=[1],
        rdts=rdts,
        output=tmp_path / "correct_fd2",
        feeddown=2,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    # Testing ------------------------------------------------------------------
    # Check output data ---
    assert len(list(tmp_path.glob("correct*"))) == 6

    # Check all found correctors ---
    # no corrections with feed-down
    assert all(df_corrections[VALUE] == 0)

    if x == 0 and y == 0:
        assert all(df_corrections_fd1[VALUE] == 0)
        assert all(df_corrections_fd2[VALUE] == 0)

    else:
        for ip in correct_ips:
            normal_oct_mask = (df_corrections[STRENGTH] == "K3L") & (df_corrections[IP] == ip)
            skew_oct_mask = (df_corrections[STRENGTH] == "K3SL") & (df_corrections[IP] == ip)
            dodecapole_error_sum = error_value * n_magnets * n_sides
            norm_oct_corr_fd1 = sum(df_corrections_fd1.loc[normal_oct_mask, VALUE])
            skew_oct_corr_fd1 = sum(df_corrections_fd1.loc[skew_oct_mask, VALUE])
            assert abs(norm_oct_corr_fd1 + x * dodecapole_error_sum) < EPS
            assert abs(skew_oct_corr_fd1 + y * dodecapole_error_sum) < EPS

            norm_oct_corr_fd2 = sum(df_corrections_fd2.loc[normal_oct_mask, VALUE])
            skew_oct_corr_fd2 = sum(df_corrections_fd2.loc[skew_oct_mask, VALUE])
            assert abs(norm_oct_corr_fd2 + 0.5 * (x**2 - y**2) * dodecapole_error_sum) < EPS
            assert abs(skew_oct_corr_fd2 + x * y * dodecapole_error_sum) < EPS


@pytest.mark.parametrize('corrector', ("a5", "b5", "a6", "b6"))
@pytest.mark.parametrize('x', (2, 0))
@pytest.mark.parametrize('y', (2, 1.5, 0))
def test_correct_via_feeddown(tmp_path: Path, x: float, y: float, corrector: str):
    """Test correct RDT via feeddown from higher order corrector.
    In this example: Use normal and skew deca- and dodecapole correctors
    to correct for normal octupole errors (which make it easy to
    just sum up over both sides).
    """
    # Parameters -----------------------------------------------------------
    accel = 'hllhc'

    correct_ips = (1, 3)
    error_value = 2
    n_magnets = 4
    n_ips = 4
    n_sides = 2

    # Setup ----------------------------------------------------------------
    optics = generate_pseudo_model(
        accel=accel, n_ips=n_ips, n_magnets=n_magnets, x=x, y=y)
    errors = generate_errortable(
        index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
    )
    errors["K3L"] = error_value  # octupole errors

    # Correction ---------------------------------------------------------------
    rdts = {"f4000": [corrector]}
    _, df_corrections = irnl_correct(
        accel=accel,
        optics=[optics],
        errors=[errors],
        beams=[1],
        rdts=rdts,
        output=tmp_path / "correct",
        feeddown=0,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    assert len(df_corrections.index) == len(correct_ips) * n_sides
    assert all(df_corrections[FIELD] == corrector)

    coeff = {"a5": y, "b5": x, "a6": y*x, "b6": 0.5*(x**2 - y**2)}[corrector]
    if coeff == 0:
        # No Feed-down possible
        assert all(df_corrections[VALUE] < EPS)
    else:
        # as beta cancels out (and is 1 anyway)
        error_strengths = n_sides * n_magnets * error_value
        for ip in correct_ips:
            mask = df_corrections[IP] == ip
            corrector_strengths = coeff * sum(df_corrections.loc[mask, VALUE])
            assert abs(corrector_strengths + error_strengths) < EPS  # compensation of RDT
