from pathlib import Path

import numpy as np
import pytest

from irnl_rdt_correction.irnl_rdt_correction import main as irnl_correct
from tests.helpers import (
    generate_pseudo_model, generate_errortable, get_some_magnet_names, IP, VALUE, EPS
)


def test_dual_optics(tmp_path: Path):
    """Test that given two different optics, an approximative solution
    will be found."""
    # Parameters -----------------------------------------------------------
    accel = 'hllhc'

    correct_ips = (1, 3)
    n_magnets = 4
    n_ips = 4
    n_sides = 2

    # Setup ----------------------------------------------------------------
    beta = 2
    error_value = 2
    twiss1 = generate_pseudo_model(
        accel=accel, n_ips=n_ips, n_magnets=n_magnets, betax=beta, betay=beta)
    errors1 = generate_errortable(
        index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
        value=error_value,
    )

    # Optics 2
    beta2 = 4
    error_value2 = 3 * error_value
    twiss2 = generate_pseudo_model(
        accel=accel, n_ips=n_ips, n_magnets=n_magnets, betax=beta2, betay=beta2)
    errors2 = generate_errortable(
        index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
        value=error_value2,
    )

    # Correction ---------------------------------------------------------------
    rdt = "f4000"

    # The corrector values in this example are not uniquely defined
    # so these methods will fail:
    for solver in ["inv", "linear"]:
        with pytest.raises(np.linalg.LinAlgError):
            _, df_corrections = irnl_correct(
                accel=accel,
                twiss=[twiss1, twiss2],
                errors=[errors1, errors2],
                beams=[1, 1],
                rdts=[rdt, ],
                ips=correct_ips,
                ignore_missing_columns=True,
                iterations=1,
                solver=solver
            )

    # Best approximation for corrector values, via least-squares:
    _, df_corrections = irnl_correct(
        accel=accel,
        twiss=[twiss1, twiss2],
        errors=[errors1, errors2],
        beams=[1, 1],
        rdts=[rdt, ],
        output=tmp_path / "correct_dual",
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
        solver="lstsq",
    )

    # as beta cancels out:
    error_strengths1 = n_sides * n_magnets * error_value
    error_strengths2 = n_sides * n_magnets * error_value2

    # build the equation system manually, and solve with least square
    # (basically what the correction should do):
    exp_x = (int(rdt[1]) + int(rdt[2])) / 2
    exp_y = (int(rdt[2]) + int(rdt[3])) / 2
    b1 = beta**(exp_x+exp_y)
    b2 = beta2**(exp_x+exp_y)
    dual_correction = np.linalg.lstsq(np.array([[b1, b1], [b2, b2]]),
                                      np.array([-b1*error_strengths1, -b2*error_strengths2]))[0]

    assert all(np.abs(dual_correction) > 0)  # just for safety, that there is a solution

    for ip in correct_ips:
        mask = df_corrections[IP] == ip
        assert all(np.abs((df_corrections.loc[mask, VALUE] - dual_correction)) < EPS)


def test_dual_optics_rdts(tmp_path: Path):
    """Test calculations given two different optics and different RDTs."""
    # Parameters -----------------------------------------------------------
    accel = 'hllhc'

    correct_ips = (1, 3)
    n_magnets = 4
    n_ips = 4
    n_sides = 2

    # Setup ----------------------------------------------------------------
    rdt1 = "f4000"
    beta = 2
    error_value = 2
    twiss1 = generate_pseudo_model(
        accel=accel, n_ips=n_ips, n_magnets=n_magnets, betax=beta, betay=beta)
    errors1 = generate_errortable(
        index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
        value=error_value,
    )

    # Optics that require same strengths with rdt2
    rdt2 = "f2002"
    beta2 = 4
    error_value2 = error_value
    twiss2 = generate_pseudo_model(
        accel=accel, n_ips=n_ips, n_magnets=n_magnets, betax=beta2, betay=beta2)

    errors2 = generate_errortable(
        index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
        value=error_value2,
    )

    # Correction ---------------------------------------------------------------
    _, df_corrections = irnl_correct(
        accel=accel,
        twiss=[twiss1, twiss2],
        errors=[errors1, errors2],
        beams=[1, 1],
        rdts=[rdt1, ],
        rdts2=[rdt2, ],
        ips=correct_ips,
        ignore_missing_columns=True,
        iterations=1,
    )

    # as beta cancels out:
    error_strengths = n_sides * n_magnets * error_value

    for ip in correct_ips:
        mask = df_corrections[IP] == ip
        corrector_strengths = sum(df_corrections.loc[mask, VALUE])
        assert abs(corrector_strengths + error_strengths) < EPS  # compensation of RDT
