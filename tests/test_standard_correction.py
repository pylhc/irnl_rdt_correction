from pathlib import Path

import numpy as np
import pytest

from irnl_rdt_correction.constants import KEYWORD
from irnl_rdt_correction.main import irnl_rdt_correction
from tests.helpers import (
    generate_pseudo_model, generate_errortable, get_some_magnet_names,
    read_lhc_model, get_ir_magnets_mask, get_corrector_magnets_mask,
    get_opposite_sign_beam4_kl_columns,
    NAME, PLACEHOLDER, CIRCUIT, STRENGTH, IP, EPS, VALUE, MAX_N,
)


@pytest.mark.parametrize('order', range(3, MAX_N+1))  # 3 == Sextupole
@pytest.mark.parametrize('orientation', ('skew', 'normal'))
@pytest.mark.parametrize('accel', ('lhc', 'hllhc'))
def test_basic_correction(tmp_path: Path, order: int, orientation: str, accel: str):
    """Tests the basic correction functionality and performs some sanity checks.
    Operates on a pseudo-model so that the corrector values are easily known.
    Sanity Checks:
    - all correctors found
    - correctors have the correct value (as set by errors or zero)
    - all corrector circuits present in madx-script
    """
    # Parameters -----------------------------------------------------------
    if accel == 'lhc':
        if order == 5:
            pytest.skip("LHC has no decapole correctors")
        if order == 6 and orientation == 'skew':
            pytest.skip("LHC has no skew dodecapole correctors")

    orientation = "S" if orientation is "skew" else ""

    correct_ips = (1, 3)
    error_value = 2
    n_magnets = 4
    n_ips = 4

    n_correct_ips = len(correct_ips)
    n_sides = len("LR")
    n_orientation = len(["S", ""])

    # Setup ----------------------------------------------------------------
    twiss = generate_pseudo_model(accel=accel, n_ips=n_ips, n_magnets=n_magnets)
    errors = generate_errortable(index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets))
    error_component = f"K{order-1}{orientation}L"
    errors[error_component] = error_value

    if order % 2:  # order is odd -> sides have different sign in rdt
        left_hand_magnets = errors.index.str.match(r".*L\d$")
        errors.loc[left_hand_magnets, error_component] = errors.loc[left_hand_magnets, error_component] / 2  # so they don't fully compensate

    # Correction -----------------------------------------------------------
    madx_corrections, df_corrections = irnl_rdt_correction(
        accel=accel,
        twiss=[twiss],
        errors=[errors],
        beams=[1],
        output=tmp_path / "correct",
        feeddown=0,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    # Testing --------------------------------------------------------------
    # Check output data ---
    assert len(list(tmp_path.glob("correct.*"))) == 2

    # Check all found correctors ---
    if accel == 'lhc':
        assert len(df_corrections.index) == (
                n_orientation * n_sides * n_correct_ips * len("SO") +
                n_sides * n_correct_ips * len("T")
        )

    if accel == 'hllhc':
        assert len(df_corrections.index) == n_orientation * n_sides * n_correct_ips * len("SODT")

    # All circuits in madx script ---
    for circuit in df_corrections[CIRCUIT]:
        assert circuit in madx_corrections

    # Check corrector values ---
    for test_order in range(3, MAX_N+1):
        for test_orientation in ("S", ""):
            for ip in correct_ips:
                mask = (
                        (df_corrections[STRENGTH] == f"K{test_order-1}{test_orientation}L") &
                        (df_corrections[IP] == ip)
                )
                if (test_order == order) and (test_orientation == orientation):
                    if order % 2:
                        corrector_strengths = sum(df_corrections.loc[mask, VALUE])
                        assert abs(corrector_strengths) < EPS  # correctors should be equally distributed

                        corrector_strengths = -sum(df_corrections.loc[mask, VALUE].abs())
                        # as beta cancels out (and is 1 anyway)
                        error_strengths = n_magnets * error_value / 2  # account for partial compensation (from above)
                    else:
                        corrector_strengths = sum(df_corrections.loc[mask, VALUE])
                        assert all(abs(df_corrections.loc[mask, VALUE] - corrector_strengths / n_sides) < EPS)
                        # as beta cancels out (and is 1 anyway)
                        error_strengths = (n_sides * n_magnets * error_value)
                    assert abs(corrector_strengths + error_strengths) < EPS  # compensation of RDT
                else:
                    assert all(df_corrections.loc[mask, VALUE] == 0.)


@pytest.mark.parametrize('beam', (1, 2, 4))
def test_lhc_correction(tmp_path: Path, beam: int):
    """Test LHC optics with random errors assigned.
    Sanity Checks:
    - all correctors found
    - all correctors have a value
    - all corrector circuits present in madx-script
    """
    # Setup ----------------------------------------------------------------
    np.random.seed(20211108)
    twiss = read_lhc_model(beam)
    mask_ir = get_ir_magnets_mask(twiss.index)
    twiss = twiss.loc[mask_ir, :]
    correctors = twiss.index[get_corrector_magnets_mask(twiss.index)]
    correct_ips = (1, 5)
    correctors = [c for c in correctors if int(c[-1]) in correct_ips]

    errors = generate_errortable(index=twiss.index)

    # here: 2 == sextupole
    errors.loc[:, [f"K{order}{orientation}L"
                   for order in range(2, MAX_N) for orientation in ("S", "")]] = np.random.random([len(errors.index), 8])
    if beam == 4:
        negative_columns = get_opposite_sign_beam4_kl_columns(range(2, MAX_N))
        errors.loc[:, negative_columns] = -errors.loc[:, negative_columns]

    # Correction -----------------------------------------------------------
    madx_corrections, df_corrections = irnl_rdt_correction(
        accel='lhc',
        twiss=[twiss],
        errors=[errors],
        beams=[beam],
        output=tmp_path / "correct",
        feeddown=0,
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    # Testing --------------------------------------------------------------
    # Check output data ---
    assert len(list(tmp_path.glob("correct.*"))) == 2

    # All correctors present with a value ---
    assert len(df_corrections.index) == 2 * 2 * 5 - 1  # sides * ips * corrector orders - faulty MCOSX.3L1
    assert all(df_corrections[VALUE] != 0)

    found_correctors = df_corrections[NAME].to_numpy()
    for name in correctors:
        if twiss.loc[name, KEYWORD] == PLACEHOLDER:
            continue
        assert name in found_correctors

    # all corrector strengths are negative because all errors are positive (np.random.random)
    # this checks, that there is no sign-change between beam 1, 2 and 4.
    assert all(df_corrections[VALUE] < 0)

    # All circuits in madx script ---
    for circuit in df_corrections[CIRCUIT]:
        assert circuit in madx_corrections
