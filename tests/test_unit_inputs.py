from dataclasses import asdict, fields
from pathlib import Path
from pandas import DataFrame
from tfs import TfsDataFrame
import pytest
from unittest.mock import Mock

from tests.helpers import cli_args
from irnl_rdt_correction.main import irnl_rdt_correction
from irnl_rdt_correction.input_options import InputOptions, allow_commandline_and_kwargs


def test_wrong_arguments():
    with pytest.raises(TypeError) as e:
        irnl_rdt_correction(
            feddown=0,
        )
    assert "feddown" in str(e)
    
    with pytest.raises(TypeError) as e:
        irnl_rdt_correction(
            itterations=0,
        )
    assert "itterations" in str(e)


class TestInputOptions:
    def test_check_accel_invalid(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.accel = "invalid_accel"

        with pytest.raises(ValueError, match="Parameter 'accel' needs to be one of"):
            invalid_input_options.check_accel()

    def test_check_twiss_missing(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.twiss = None

        with pytest.raises(ValueError, match="Parameter 'twiss' needs to be set."):
            invalid_input_options.check_twiss()

    def test_check_twiss_invalid_type(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.twiss = ["twiss1", "twiss2", 123]

        with pytest.raises(TypeError, match="Not all elements of 'twiss' are DataFrames or paths to DataFrames!"):
            invalid_input_options.check_twiss()

    def test_check_errors_invalid_type(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.errors = ["errors1", "errors2", 123, None]

        with pytest.raises(TypeError, match="Not all elements of 'errors' are DataFrames or paths to DataFrames or None!"):
            invalid_input_options.check_errors()

    def test_check_beams_missing(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.beams = None

        with pytest.raises(ValueError, match="Parameter 'beams' needs to be set."):
            invalid_input_options.check_beams()

    def test_check_rdts_missing(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.rdts = None

        invalid_input_options.check_rdts()
        assert invalid_input_options.rdts == InputOptions.DEFAULT_RDTS[invalid_input_options.accel]

    def test_check_rdts_invalid_type(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.rdts = "invalid_rdts"

        with pytest.raises(ValueError, match="Parameter 'rdts' needs to be iterable"):
            invalid_input_options.check_rdts()

    def test_check_feeddown_invalid(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.feeddown = -1

        with pytest.raises(ValueError, match="'feeddown' needs to be a positive integer."):
            invalid_input_options.check_feeddown()

    def test_check_iterations_invalid(self, valid_input_options):
        invalid_input_options = valid_input_options
        invalid_input_options.iterations = 0

        with pytest.raises(ValueError, match="At least one iteration"):
            invalid_input_options.check_iterations()

    def test_post_init_calls_check_all(self, valid_input_options):
        class MockInputOptions(InputOptions):
            check_all = Mock()

        mock_input_options = MockInputOptions(**asdict(valid_input_options))

        mock_input_options.check_all.assert_called_once()

    def test_input_options_behaves_as_dict(self, valid_input_options):
        fields_ = list(fields(valid_input_options))
        items = list(valid_input_options.items())
        keys = list(valid_input_options.keys())
        values = list(valid_input_options.values())

        assert len(items)
        assert len(fields_) == len(items)
        assert len(keys) == len(items)
        assert len(values) == len(items)

        for keyvalue, key, value in zip(items, keys, values):
            assert getattr(valid_input_options, key) == value
            assert valid_input_options[key] == value
            assert keyvalue[0] == key
            assert keyvalue[1] == value
        
    
    def test_post_init_calls_all_checks(self, valid_input_options):
        check_functions = [fun for fun in dir(InputOptions) if fun.startswith("check_") and not fun == "check_all"]
        assert len(check_functions)

        class MockInputOptions(InputOptions):
            pass

        for check_function in check_functions:
            setattr(MockInputOptions, check_function, Mock())
            
        mock_input_options = MockInputOptions(**asdict(valid_input_options))

        for check_function in check_functions:
            getattr(mock_input_options, check_function).assert_called_once()


class TestInputWrapper:

    def test_with_input_options(self):
        in_opt = InputOptions(beams=[1,2], twiss=["test.tfs", "test1.tfs"])
        out_opt = _multi_input_fun(in_opt)
        assert in_opt is out_opt

    def test_with_commandline(self):
        with cli_args("--beams", "1", "2", "--twiss", "test.tfs", "test1.tfs"):
            out_opt = _multi_input_fun()
        assert out_opt.beams == [1, 2]
        assert out_opt.twiss == ["test.tfs", "test1.tfs"]
        assert out_opt.errors == (None, None)


    def test_with_dict(self):
        in_opt = dict(beams=[1,2], twiss=["test.tfs", "test1.tfs"], errors=["errors.tfs", "more_errors.tfs"])
        out_opt = _multi_input_fun(in_opt)
        for key, value in in_opt.items():
            assert getattr(out_opt, key) == value

    def test_with_kwargs(self):
        in_opt = dict(beams=[1,2], twiss=["test.tfs", "test1.tfs"], errors=["errors.tfs", "more_errors.tfs"])
        out_opt = _multi_input_fun(**in_opt)
        for key, value in in_opt.items():
            assert getattr(out_opt, key) == value


# Helper -----------------------------------------------------------------------

@allow_commandline_and_kwargs
def _multi_input_fun(opt):
    return opt


@pytest.fixture
def valid_input_options() -> InputOptions:
    return InputOptions(
        beams=[1, 2, 3],
        twiss=[Path("twiss.dat"), DataFrame(), TfsDataFrame()],
        errors=[Path("errors.dat"), DataFrame(), TfsDataFrame(), None],
        accel="lhc",
        rdts=["F0003", "F1002"],
        feeddown=0,
        iterations=1,
    )