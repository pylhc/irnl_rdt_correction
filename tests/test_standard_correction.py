from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from irnl_rdt_correction.constants import KEYWORD, DELTA, BETA
from irnl_rdt_correction.main import irnl_rdt_correction
from tests.helpers import (
    generate_pseudo_model, generate_errortable, get_some_magnet_names,
    read_lhc_model, get_ir_magnets_mask, get_corrector_magnets_mask,
    get_opposite_sign_beam4_kl_columns,
    NAME, PLACEHOLDER, CIRCUIT, STRENGTH, IP, EPS, VALUE, MAX_N,
    X, Y
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

    orientation = "S" if orientation == "skew" else ""

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


@pytest.mark.parametrize('accel', ('lhc', 'hllhc'))
@pytest.mark.parametrize('feeddown', (0, 2))
def test_beams_symmetries(tmp_path: Path, accel: str, feeddown: int):
    """ Very similar to the basic correction, but here we check that we actually 
    get the correct symmetries between the beams. 
    It also checks that the pre-powering of the magnets is stored with the correct sign.
    This test should fail with irnl_correction versions <= v1.0.1.
    """
    np.random.seed(20231003) 

    correct_ips = (1,)
    n_magnets = 1
    n_ips = 1

    is_lhc = accel == 'lhc'
    orientations = ("", "S")
    orders = list(range(2, 6))
    order_letters = ["S", "O", "D", "T"]

    if feeddown:
        rdts = ['F0003', 'F0003*', 'F1002', 'F1002*']
    else:
        rdts = ['F0003', 'F0003*', 'F1002', 'F1002*',
                'F1003', 'F3001', 'F4000', 'F0004',] + ([] if is_lhc else ['F0005', 'F0005*', 'F5001', 'F1005'])


    corrector_order_map = {f"{order_name}{skew}": f"K{order}{skew}L" for order, order_name in zip(orders, order_letters) for skew in orientations}  # madx order

    beam_results = {}
    
    @dataclass
    class BeamResult:
        beam: int
        twiss: pd.DataFrame
        errors: pd.DataFrame
        madx_corrections: str = None
        df_corrections: pd.DataFrame = None
        
    # Setup ----------------------------------------------------------------
    # The setup is designed such that without feeddown, all corrections 
    # yield the same values, which are n_magnets * 1.5 on the left and n_magents * 2 on the right.
    # With feeddown only a3 and b3 are corrected:
    # a3 = a3 + x*a4 + y*b4 + xy*b5 + 1/2(x**2 - y**2)*a5
    # b3 = b3 + x*b4 - y*b4 - xy*a5 + 1/2(x**2 - y**2)*b5
    # as x == y == 1
    # a3 = 4 * n_magnets * 1.5
    # b3 = 0

    twiss_0 = generate_pseudo_model(accel=accel, n_ips=n_ips, n_magnets=n_magnets, betax=1, betay=1)
    errors_0 = generate_errortable(index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets), max_order=0)
    
    # here: 2 == sextupole
    kl_columns = [f"K{order}{orientation}L" for order in orders for orientation in orientations]
    
    twiss_0.loc[:, kl_columns]  = 0
    errors_0.loc[:, kl_columns]  = 0

    # orbit at corrector should not matter, as we don't use feed-down to correct here)
    # twiss_0.loc[:, [X, Y]] = 0.5
    # errors_0.loc[:, [f"{DELTA}{X}", f"{DELTA}{Y}"]] = 0.5

    # Pre-Power corrector magnets, to test this does not cause problems (<=v1.0.1 does) 
    # only power the field that will be modified by the correction, 
    # e.g. normal quadrupole field for normal quadrupole correctors
    if feeddown == 0:
        for magnet_order, kl in corrector_order_map.items():
            if magnet_order in ("D", "DS") and is_lhc:
                continue
            corrector_magnets = twiss_0.index.str.match(rf"MC{magnet_order}X") 
            twiss_0.loc[corrector_magnets, kl] = 1

    # Power other magnets and assign errors
    non_corrector_magnets = twiss_0.index.str.match(r"M.\.")  # M[A-Z]. is created above
    twiss_0.loc[non_corrector_magnets, kl_columns] = 1
    twiss_0.loc[non_corrector_magnets, [X, Y]] = 0.5
    
    non_corrector_magnets = errors_0.index.str.match(r"M.\.")  # M[A-Z]. is created above
    errors_0.loc[non_corrector_magnets, kl_columns] = 1
    errors_0.loc[non_corrector_magnets, [f"{DELTA}{X}", f"{DELTA}{Y}"]] = 0.5

    # modify the left side, so they don't fully compensate for even (madx)-orders
    left_hand_magnets = twiss_0.index.str.match(r".*L\d$")
    twiss_0.loc[left_hand_magnets, f"{BETA}{Y}"] = 2 * twiss_0.loc[left_hand_magnets, f"{BETA}{Y}"]

    left_hand_magnets = errors_0.index.str.match(r".*L\d$")
    errors_0.loc[left_hand_magnets, kl_columns] = errors_0.loc[left_hand_magnets, kl_columns] / 2  

    # Pre-calculate the integral based on this setup. 
    integral_left = 1.5 * n_magnets
    integral_right = 2.0 * n_magnets

    # Correction -----------------------------------------------------------
    for beam in (1, 2, 4):
        # Create twiss per beam based on twiss_0 and the symmetries involved, 
        # using the conventions for twiss and errors as implemented in MAD-X.
        # ------- Reminder of MAD-X conventions ----------
        #           Orbit:                        antisymmetic K:
        # twiss:     X B1 =  X B2 =  -X B4        K B1 = -K B2 =  K B4 (i.e. K B1 = -K B4)
        # errors:   DX B1 = DX B2 = -DX B4        K B1 =  K B2 = -K B4 
        #
        # BUT ALL CORRECTIONS SHOULD BE THE SAME, 
        # as the the correction circuits are the same between all beams!

        twiss = twiss_0.copy()
        errors = errors_0.copy()

        # Transform symmetries ----
        negative_columns = get_opposite_sign_beam4_kl_columns(orders)
        if beam in (2, 4):
            twiss.loc[:, negative_columns] = -twiss.loc[:, negative_columns]
        
        if beam == 4:
            twiss.loc[:, X] = -twiss.loc[:, X]
            errors.loc[:, f"{DELTA}{X}"] = -errors.loc[:, f"{DELTA}{X}"]
            errors.loc[:, negative_columns] = -errors.loc[:, negative_columns]

        beam_results[beam] = BeamResult(
            beam=beam,
            twiss=twiss,
            errors=errors,
        )

        # Calculate corrections ---
        beam_results[beam].madx_corrections, beam_results[beam].df_corrections = irnl_rdt_correction(
            accel=accel,
            twiss=[twiss],
            errors=[errors],
            beams=[beam],
            rdts=rdts,
            output=tmp_path / "correct",
            feeddown=feeddown,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
            # ignore_corrector_settings = True,
        )

    # Testing --------------------------------------------------------------
    # Check output data ---
    
    # DEBUGGING: ---------------------------------
    # for i in (1, 2, 4):
    #     print(f"\n--- BEAM {i} ---")
    #     print(beam_results[i].madx_corrections)
    # --------------------------------------------

    eps = 1e-14

    # Comapre values between beams (this should always be true!!)
    df_beam1 = beam_results[1].df_corrections
    for beam in (2, 4):
        df = beam_results[beam].df_corrections
        assert np.allclose(df[VALUE], df_beam1[VALUE], atol=eps)

    # Check values per beam (this depends a bit on the setup above)
    for beam in (1, 2, 4):
        df = beam_results[beam].df_corrections
        assert len(df.index) == len(rdts)
        left_correctors = df[CIRCUIT].str.match(r".*L\d$")
        right_correctors = df[CIRCUIT].str.match(r".*R\d$")
        assert left_correctors.sum() == len(rdts) * n_ips / 2
        assert right_correctors.sum() == len(rdts) * n_ips / 2

        if feeddown == 0:
            assert np.allclose(df.loc[left_correctors, VALUE], -integral_left, atol=eps, rtol=0)
            assert np.allclose(df.loc[right_correctors, VALUE], -integral_right, atol=eps, rtol=0)
        else:
            b3_correctors = df[CIRCUIT].str.match(r"KCSX.*")
            a3_correctors = df[CIRCUIT].str.match(r"KCSSX.*")

            assert np.allclose(df.loc[left_correctors & a3_correctors, VALUE], -integral_left * 4, atol=eps, rtol=0)
            assert np.allclose(df.loc[right_correctors & a3_correctors, VALUE], -integral_right * 4, atol=eps, rtol=0)
            
            assert np.allclose(df.loc[b3_correctors, VALUE], 0, atol=eps, rtol=0)
