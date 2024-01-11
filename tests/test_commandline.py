import subprocess
import sys
from pathlib import Path

import pytest
import tfs

from tests.helpers import (CIRCUIT, EPS, IP, MAX_N, STRENGTH, VALUE, generate_errortable,
                           generate_pseudo_model, get_some_magnet_names)


def test_package_execution_no_arguments():
    """ Tests if the package can be run as a module. 
    Without arguments it should abort with an error message stating required arguments. """
    # Run the package as a module using subprocess
    result = subprocess.run([sys.executable, "-m", "irnl_rdt_correction"], capture_output=True, text=True)

    # Check if the command executed with sysexit (exit code 2)
    assert result.returncode == 2

    # Check for expected output
    expected_output = "error: the following arguments are required: --beams, --twiss"
    assert expected_output in result.stderr


def test_package_execution_fix_v1_1_2():
    """ Tests if the package can be run as a module. This failed in <1.1.2 with an import error. """
    # Run the package as a module using subprocess
    result = subprocess.run([sys.executable, "-m", "irnl_rdt_correction"], capture_output=True, text=True)
    
    # Check if the command executed not with an import error (exit code 1)
    assert result.returncode != 1

    # Check for a module not found error
    not_expected_output = "ModuleNotFoundError: No module named 'utilities'"  
    assert not_expected_output not in result.stderr


@pytest.mark.parametrize('accel', ('lhc', 'hllhc'))
def test_basic_correction(tmp_path: Path, accel: str):
    """Tests the basic correction functionality and performs some sanity checks.
    Same as in tets_standard_correction, but less testcases and called from commandline.

    Operates on a pseudo-model so that the corrector values are easily known.
    Sanity Checks:
    - all correctors found
    - correctors have the correct value (as set by errors or zero)
    - all corrector circuits present in madx-script
    """
    # Parameters -----------------------------------------------------------
    order = 4
    orientation = ""

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
    
    twiss_path = tmp_path / "twiss_input.tfs"
    errors_path = tmp_path / "errors_input.tfs"
    result_path = tmp_path / "correct"

    tfs.write(twiss_path, twiss, save_index="NAME")
    tfs.write(errors_path, errors, save_index="NAME")

    # Correction -----------------------------------------------------------
    subprocess.run([sys.executable, "-m", "irnl_rdt_correction",
     "--accel", accel, 
    "--twiss", twiss_path,
    "--errors", errors_path,
    "--beams", "1",
    "--output", result_path,
    "--feeddown", "0",
    "--ips", *[str(ip) for ip in correct_ips],
    "--ignore_missing_columns",
    "--iterations", "1",
    ], 
    capture_output=True, text=True
    )

    madx_corrections = result_path.with_suffix(".madx").read_text()
    df_corrections = tfs.read(result_path.with_suffix(".tfs"))

    # Testing --------------------------------------------------------------
    # Same as in test_standard_correction - TODO: make function?
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
