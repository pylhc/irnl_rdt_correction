from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from irnl_rdt_correction.constants import BETA
from irnl_rdt_correction.main import irnl_rdt_correction
from tests.helpers import (
    generate_pseudo_model, get_corrector_magnets_mask, get_some_magnet_names,
    generate_errortable, VALUE, IP, EPS
)


def test_different_rdts(tmp_path: Path):
    """Test that different RDTs can be corrected and only their correctors
    are returned. Also checks that the corrector values are varying between RDTs
    when they should. Octupole RDTs are used for this example.
    """
    # Parameters -----------------------------------------------------------
    accel = 'lhc'

    correct_ips = (1, 3)
    error_value = 2
    n_magnets = 4
    n_ips = 4

    # Setup ----------------------------------------------------------------
    twiss = generate_pseudo_model(accel=accel, n_ips=n_ips, n_magnets=n_magnets)

    # use different beta for correctors to avoid beta cancellation
    # so that different RDTs give different corrector strengths
    correctors_mask = get_corrector_magnets_mask(twiss.index)
    twiss.loc[correctors_mask, f"{BETA}Y"] = 3

    errors = generate_errortable(index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets))
    errors["K3L"] = error_value

    # Correction -----------------------------------------------------------
    _, df_corrections_f4000 = irnl_rdt_correction(
        accel=accel,
        twiss=[twiss],
        errors=[errors],
        beams=[1],
        rdts=["f4000",],
        output=tmp_path / "correct4000",
        feeddown=0,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    _, df_corrections_f2200 = irnl_rdt_correction(
        accel=accel,
        twiss=[twiss],
        errors=[errors],
        beams=[1],
        rdts=["f2200",],
        output=tmp_path / "correct2200",
        feeddown=0,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    _, df_corrections_f2002 = irnl_rdt_correction(
        accel=accel,
        twiss=[twiss],
        errors=[errors],
        beams=[1],
        rdts=["f2002", ],
        output=tmp_path / "correct2002",
        feeddown=0,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    # Testing --------------------------------------------------------------
    # Check output data ---
    assert len(list(tmp_path.glob("correct*"))) == 6

    # Check all found correctors ---
    # only octupole correctors should be present
    for correction in (df_corrections_f4000, df_corrections_f2200, df_corrections_f2002):
        assert len(correction.index) == 4
        assert all(correction['order'] == 4)

    # f4000 and f2200 should give same values for correction
    assert_frame_equal(df_corrections_f4000, df_corrections_f2200)

    # f4000 and f2002 should give different values for correction
    with pytest.raises(AssertionError):
        assert_series_equal(df_corrections_f4000[VALUE], df_corrections_f2002[VALUE])

    # frames are equal apart from value, though
    non_val_columns = [col for col in df_corrections_f2200.columns if col != VALUE]
    assert_frame_equal(df_corrections_f4000[non_val_columns], df_corrections_f2002[non_val_columns])


def test_switched_beta():
    """Test using the special RDTs* where the beta-exponents are switched."""
    # Parameters -----------------------------------------------------------
    accel = 'hllhc'

    correct_ips = (1, 3)
    n_magnets = 4
    n_ips = 4
    n_sides = 2

    # Setup ----------------------------------------------------------------
    beta = 2
    error_value = 2
    twiss = generate_pseudo_model(
        accel=accel, n_ips=n_ips, n_magnets=n_magnets, betax=beta, betay=beta)
    errors = generate_errortable(
        index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
        value=error_value,
    )

    # Correction ---------------------------------------------------------------
    _, df_corrections = irnl_rdt_correction(
        accel=accel,
        twiss=[twiss, ],
        errors=[errors, ],
        beams=[1, ],
        rdts=["f4000", ],
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    _, df_corrections_switched = irnl_rdt_correction(
        accel=accel,
        twiss=[twiss, ],
        errors=[errors, ],
        beams=[1, ],
        rdts=["f0004*", ],  # only for testing purposes use this RDT
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    # as beta cancels out:
    error_strengths = n_sides * n_magnets * error_value

    for ip in correct_ips:
        mask = df_corrections_switched[IP] == ip
        corrector_strengths_switched = sum(df_corrections_switched.loc[mask, VALUE])
        assert abs(corrector_strengths_switched + error_strengths) < EPS  # compensation of RDT
    assert_frame_equal(df_corrections, df_corrections_switched)
