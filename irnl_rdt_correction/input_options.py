"""
Input Options
-------------

Handles the input parameters, contains their default values
and checks for their validity.
"""
import argparse
from dataclasses import dataclass, fields
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Sized, Union, Tuple

import pandas as pd
import tfs

from irnl_rdt_correction.constants import EXT_TFS, EXT_MADX, StrOrPathOrDataFrame, StrOrPathOrDataFrameOrNone
from irnl_rdt_correction.equation_system import SOLVER_MAP
from irnl_rdt_correction.utilities import list2str

LOG = logging.getLogger(__name__)


# Parser -----------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beams",
        dest="beams",
        type=int,
        nargs="+",
        help="Which beam the files come from (1, 2 or 4)",
        required=True,
    )
    parser.add_argument(
        "--twiss",
        dest="twiss",
        nargs="+",
        help="Path(s) to twiss file(s), in the format of MAD-X `twiss` output. "
             "Defines which elements to correct for,"
             "meaning all given elements contribute to the correction!",
        required=True,
    )
    parser.add_argument(
        "--errors",
        dest="errors",
        nargs="+",
        help="Path(s) to error file(s), in the format of MAD-X `esave` output.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        help=("Path to write command and tfs_df file. "
              f"Extension (if given) is ignored and replaced with {EXT_TFS} and {EXT_MADX} "
              "for the Dataframe and the command file respectively."
              ),
    )
    parser.add_argument(
        "--rdts",
        dest="rdts",
        nargs="+",
        help=("RDTs to correct."
              " Format: 'Fjklm'; or 'Fjklm*'"
              " to correct for this RDT in Beam 2 using"
              " beta-symmetry (jk <-> lm)."),
    )
    parser.add_argument(
        "--rdts2",
        dest="rdts2",
        nargs="+",
        help=("RDTs to correct for second beam/file, if different from first."
              " Same format rules as for `rdts`."),
    )
    parser.add_argument(
        "--accel",
        dest="accel",
        type=str.lower,
        choices=list(InputOptions.DEFAULT_RDTS.keys()),
        default=InputOptions.accel,
        help="Which accelerator we have.",
    )
    parser.add_argument(
        "--feeddown",
        dest="feeddown",
        type=int,
        help="Order of Feeddown to include.",
        default=InputOptions.feeddown,
    )
    parser.add_argument(
        "--ips",
        dest="ips",
        nargs="+",
        help="In which IPs to correct.",
        type=int,
        default=list(InputOptions.ips),
    )
    parser.add_argument(
        "--solver",
        dest="solver",
        help="Solving method to use.",
        type=str.lower,
        choices=list(SOLVER_MAP.keys()),
        default=InputOptions.solver,
    )
    parser.add_argument(
        "--update_optics",
        dest="update_optics",
        type=bool,
        help=("Sorts the RDTs to start with the highest order and updates the "
              "corrector strengths in the optics after calculation, so the "
              "feeddown to lower order correctors is included."
              ),
        default=InputOptions.update_optics
    )
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        help=("Reiterate correction, "
              "starting with the previously calculated values."),
        default=InputOptions.iterations
    )
    parser.add_argument(
        "--ignore_corrector_settings",
        dest="ignore_corrector_settings",
        help=("Ignore the current settings of the correctors. If this is not set "
              "the corrector values of the optics are used as initial conditions."),
        action="store_true",
    )
    parser.add_argument(
        "--ignore_missing_columns",
        dest="ignore_missing_columns",
        help=("If set, missing strength columns in any of the input files "
              "are assumed to be zero, instead of raising an error."),
        action="store_true",
    )
    return parser

# InputOptions and Defaults ---------------------------------------------------------------

@dataclass
class InputOptions:
    """ DataClass to store the input options.
    On creation it asserts that the input parameters make sense and adds what's missing.
    Checks include:
        - Set defaults (see ``DEFAULTS``) if option not given.
        - Check accelerator name is valid
        - Set default RDTs if not given (see ``DEFAULT_RDTS``)
        - Check required parameters are present (twiss, errors, beams, rdts)
        - Check feeddown and iterations

    """
    DEFAULT_RDTS = {
        'lhc': ('F0003', 'F0003*',  # correct a3 errors with F0003
                'F1002', 'F1002*',  # correct b3 errors with F1002
                'F1003', 'F3001',  # correct a4 errors with F1003 and F3001
                'F4000', 'F0004',  # correct b4 errors with F4000 and F0004
                'F6000', 'F0006',  # correct b6 errors with F6000 and F0006
                ),
        'hllhc': ('F0003', 'F0003*',  # correct a3 errors with F0003
                'F1002', 'F1002*',  # correct b3 errors with F1002
                'F1003', 'F3001',  # correct a4 errors with F1003 and F3001
                'F0004', 'F4000',  # correct b4 errors with F0004 and F4000
                'F0005', 'F0005*',  # correct a5 errors with F0005
                'F5000', 'F5000*',  # correct b5 errors with F5000
                'F5001', 'F1005',  # correct a6 errors with F5001 and F1005
                'F6000', 'F0006',  # correct b6 errors with F6000 and F0006
                ),
    }

    beams: Sequence[int]
    twiss: Sequence[StrOrPathOrDataFrame]
    errors: Sequence[StrOrPathOrDataFrameOrNone] = None
    rdts: Sequence[str] = None
    rdts2: Sequence[str] = None
    accel: str = 'lhc'
    feeddown: int = 0
    ips: Sequence[int] = (1, 2, 5, 8)
    solver: str ='lstsq'
    update_optics: bool = True
    iterations: int = 1
    ignore_corrector_settings: bool = False
    ignore_missing_columns: bool = False
    output: str = None

    def __post_init__(self):
        self.check_all()

    def __getitem__(self, item):
        return getattr(self, item)
    
    @classmethod
    def keys(cls):
        return (f.name for f in fields(cls))

    def values(self):
        return (getattr(self, f.name) for f in fields(self))
    
    def items(self):
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def check_all(self):
        self.check_accel()
        self.check_twiss()
        self.check_errors()
        self.check_beams()
        self.check_rdts()
        self.check_feeddown()
        self.check_iterations()

    def check_accel(self):
        if self.accel not in self.DEFAULT_RDTS:
            raise ValueError(f"Parameter 'accel' needs to be one of '{list2str(list(self.DEFAULT_RDTS.keys()))}' "
                            f"but was '{self.accel}' instead.")

    def check_twiss(self):
        if self.twiss is None:
            raise ValueError("Parameter 'twiss' needs to be set.")

        self._check_iterable('twiss') 
        for element in self.twiss:
            if not isinstance(element, (str, Path, pd.DataFrame, tfs.TfsDataFrame)):
                raise TypeError(f"Not all elements of 'twiss' are DataFrames or paths to DataFrames!")
    
    def check_errors(self):
        if self.errors is None:
            self.errors = tuple([None] * len(self.twiss))
            return

        self._check_iterable('errors')
        for element in self.errors:
            if not isinstance(element, (str, Path, pd.DataFrame, tfs.TfsDataFrame, type(None))):
                raise TypeError(f"Not all elements of 'errors' are DataFrames or paths to DataFrames or None!")
    
    def check_beams(self):
        if self.beams is None:
            raise ValueError("Parameter 'beams' needs to be set.")
        self._check_iterable('beams') 
    
    def check_rdts(self):
        if self.rdts is None:
            self.rdts = self.DEFAULT_RDTS[self.accel]
        else:
            self._check_iterable('rdts')
    
    def check_feeddown(self):
        if self.feeddown < 0 or not (self.feeddown == int(self.feeddown)):
            raise ValueError("'feeddown' needs to be a positive integer.")

    def check_iterations(self):
        if self.iterations < 1:
            raise ValueError("At least one iteration (see: 'iterations') needs to "
                            "be done for correction.")

    def _check_iterable(self, name):
        inputs = getattr(self, name)
        if isinstance(inputs, str) or not isinstance(inputs, (Iterable, Sized)):
            raise ValueError(f"Parameter '{name}' needs to be iterable, "
                             f"even if only of length 1. Instead was '{inputs}'.")

    @classmethod
    def from_args_or_dict(cls, opt: Optional[Union[dict, 'InputOptions']] = None) -> 'InputOptions':
        """Create an InputOptions instance from the given dictionary.
        If the input is empty, arguments will be parsed from commandline.

        Args:
            opt (Union[dict, DotDict]): Function options in dictionary format.
                                    Description of the arguments are given in
                                    :func:`irnl_rdt_correction.main.irnl_rdt_correction`.
                                    Optional, if not given parses commandline args

        Returns:
            InputOptions: (Parsed and) checked options.
        """
        if isinstance(opt, InputOptions):
            return opt

        if opt is None or not len(opt):
            parser = get_parser()
            opt = vars(parser.parse_args())
            
        return cls(**opt)


def allow_commandline_and_kwargs(func):
    """ Decorator to allow a function to take options from the commandline
    or via kwargs, or given an InputOptions instance. 
    """
    def wrapper(opt: Optional[Union[InputOptions, dict]] = None, **kwargs) -> Tuple[str, tfs.TfsDataFrame]:
        if not isinstance(opt, InputOptions):
            if opt is None:
                opt = kwargs
            opt = InputOptions.from_args_or_dict(opt)
        return func(opt)
    return wrapper
