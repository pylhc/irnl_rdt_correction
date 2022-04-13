import pytest
from pandas.testing import assert_frame_equal

from irnl_rdt_correction.equation_system import get_integral_sign
from irnl_rdt_correction.io_handling import maybe_switch_signs
from irnl_rdt_correction.main import irnl_rdt_correction
from irnl_rdt_correction.rdt_handling import IRCorrector, RDT
from irnl_rdt_correction.utilities import list2str, Optics
from tests.helpers import (
    ABC, generate_pseudo_model, get_opposite_sign_beam4_kl_columns, generate_errortable,
    MAX_N,
)


def test_get_integral_sign():
    for n in range(10):
        assert get_integral_sign(n, "R") == (-1)**n
        assert get_integral_sign(n, "L") == 1


def test_list_to_str():
    assert ABC == "".join(list2str(list(ABC)).replace(" ", "").replace("'", "").replace('"', "").split(','))


def test_wrong_arguments():
    with pytest.raises(AttributeError) as e:
        irnl_rdt_correction(
            feddown=0,
            itterations=1,
        )
    assert "feddown" in str(e)
    assert "itterations" in str(e)


@pytest.mark.parametrize('beam', (1, 2, 4))
def test_switch_signs(beam: int):
    all_k = [f"K{order}{orientation}L" for order in range(2, MAX_N) for orientation in ("S", "")]
    twiss = generate_pseudo_model(n_ips=1, n_magnets=10, accel='lhc', x=10, y=5)
    twiss[all_k] = 1

    errors = generate_errortable(index=twiss.index, value=2.)

    # make copies as it switches in place
    twiss_switch = twiss.copy()
    errors_switch = errors.copy()
    maybe_switch_signs(Optics(beam=beam, twiss=twiss_switch, errors=errors_switch))

    if beam != 2:
        assert_frame_equal(twiss, twiss_switch)
        assert_frame_equal(errors, errors_switch)
    else:
        switch_col_optics_mask = twiss.columns.isin(["X"])
        assert_frame_equal(twiss.loc[:, switch_col_optics_mask], -twiss_switch.loc[:, switch_col_optics_mask])
        assert_frame_equal(twiss.loc[:, ~switch_col_optics_mask], twiss_switch.loc[:, ~switch_col_optics_mask])

        switch_col_errors_mask = errors.columns.isin(["DX"] + get_opposite_sign_beam4_kl_columns(range(MAX_N)))
        assert_frame_equal(errors.loc[:, switch_col_errors_mask], -errors_switch.loc[:, switch_col_errors_mask])
        assert_frame_equal(errors.loc[:, ~switch_col_errors_mask], errors_switch.loc[:, ~switch_col_errors_mask])


def test_ircorrector_class():
    # Test Corrector
    a5_corrector_L1 = IRCorrector(field_component="a5", accel="lhc", ip=1, side="L")

    # Test Equality
    assert a5_corrector_L1 == IRCorrector(field_component="a5", accel="lhc", ip=1, side="L")
    assert a5_corrector_L1 != IRCorrector(field_component="a4", accel="lhc", ip=1, side="L")

    # Test > and < per order (important for feed-down!)
    assert a5_corrector_L1 > IRCorrector(field_component="a4", accel="lhc", ip=1, side="L")
    assert a5_corrector_L1 > IRCorrector(field_component="a4", accel="lhc", ip=2, side="R")
    assert a5_corrector_L1 > IRCorrector(field_component="b4", accel="lhc", ip=1, side="L")
    assert a5_corrector_L1 < IRCorrector(field_component="a6", accel="lhc", ip=1, side="L")
    assert a5_corrector_L1 < IRCorrector(field_component="b6", accel="lhc", ip=1, side="L")
    assert a5_corrector_L1 < IRCorrector(field_component="b6", accel="lhc", ip=8, side="R")

    # These ones are arbitrary, just to allow sorting/make sorting unique
    assert a5_corrector_L1 > IRCorrector(field_component="b5", accel="lhc", ip=1, side="L")
    assert a5_corrector_L1 < IRCorrector(field_component="a5", accel="lhc", ip=1, side="R")
    assert a5_corrector_L1 < IRCorrector(field_component="a5", accel="lhc", ip=2, side="L")


def test_ircorrector_accel():
    a4_corrector_L1 = IRCorrector(field_component="a4", accel="lhc", ip=1, side="L")
    assert "F" not in a4_corrector_L1.name

    a4_corrector_L1_hllhc = IRCorrector(field_component="a4", accel="hllhc", ip=1, side="L")
    assert "F" in a4_corrector_L1_hllhc.name
    assert a4_corrector_L1_hllhc.name.startswith("MCOS")
    assert a4_corrector_L1 != a4_corrector_L1_hllhc

    assert IRCorrector(field_component="a4", accel="lhc", ip=2, side="L") == IRCorrector(field_component="a4", accel="hllhc", ip=2, side="L")

    assert IRCorrector(field_component="b2", accel="hllhc", ip=1, side="L").name.startswith("MCQ")
    assert IRCorrector(field_component="a2", accel="hllhc", ip=1, side="L").name.startswith("MCQS")

    assert IRCorrector(field_component="b3", accel="hllhc", ip=1, side="L").name.startswith("MCS")
    assert IRCorrector(field_component="a3", accel="hllhc", ip=1, side="L").name.startswith("MCSS")

    assert IRCorrector(field_component="b4", accel="hllhc", ip=1, side="L").name.startswith("MCO")
    assert IRCorrector(field_component="a4", accel="hllhc", ip=1, side="L").name.startswith("MCOS")

    assert IRCorrector(field_component="b5", accel="hllhc", ip=1, side="L").name.startswith("MCD")
    assert IRCorrector(field_component="a5", accel="hllhc", ip=1, side="L").name.startswith("MCDS")

    assert IRCorrector(field_component="b6", accel="hllhc", ip=1, side="L").name.startswith("MCT")
    assert IRCorrector(field_component="a6", accel="hllhc", ip=1, side="L").name.startswith("MCTS")


def test_rdt_init():
    jklm = (1, 2, 3, 4)

    rdt = RDT(name=f"f{''.join(str(ii) for ii in jklm)}")
    assert rdt.order == sum(jklm)
    assert rdt.jklm == jklm
    assert rdt.j == jklm[0]
    assert rdt.k == jklm[1]
    assert rdt.l == jklm[2]
    assert rdt.m == jklm[3]
    assert not rdt.swap_beta_exp
    assert RDT("f1001*").swap_beta_exp


def test_rdt_equality():
    assert RDT("f2110") == RDT("f2110")
    assert RDT("f2110") != RDT("f2110*")


def test_rdt_sortable():
    # sortable by order
    assert RDT("f1001") < RDT("f2001")
    assert RDT("f1003") > RDT("f2001")

    # arbitrary (so sorting is unique)
    assert RDT("f1001") > RDT("f2000")
    assert RDT("f3002") < RDT("f2003")
    assert RDT("f2110") < RDT("f2110*")
    assert RDT("f1001*") > RDT("f1001")
