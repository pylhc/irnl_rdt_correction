"""
Input Options
-------------

Handles the input parameters, contains their default values
and checks for their validity.
"""
import argparse
import logging
from pathlib import Path
from typing import Iterable, Sized, Union

import pandas as pd
import tfs

from irnl_rdt_correction.constants import EXT_TFS, EXT_MADX
from irnl_rdt_correction.equation_system import SOLVER_MAP
from irnl_rdt_correction.utilities import list2str, DotDict

LOG = logging.getLogger(__name__)


# Default Values ---------------------------------------------------------------

DEFAULTS = {'feeddown': 0,
            'ips': [1, 2, 5, 8],
            'accel': 'lhc',
            'solver': 'lstsq',
            'update_optics': True,
            'iterations': 1,
            'ignore_corrector_settings': False,
            'rdts2': None,
            'ignore_missing_columns': False,
            'output': None,
            }

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


# Parser -----------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
        required=True,
    )
    parser.add_argument(
        "--beams",
        dest="beams",
        type=int,
        nargs="+",
        help="Which beam the files come from (1, 2 or 4)",
        required=True,
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
        choices=list(DEFAULT_RDTS.keys()),
        default=DEFAULTS['accel'],
        help="Which accelerator we have.",
    )
    parser.add_argument(
        "--feeddown",
        dest="feeddown",
        type=int,
        help="Order of Feeddown to include.",
        default=DEFAULTS['feeddown'],
    )
    parser.add_argument(
        "--ips",
        dest="ips",
        nargs="+",
        help="In which IPs to correct.",
        type=int,
        default=DEFAULTS['ips'],
    )
    parser.add_argument(
        "--solver",
        dest="solver",
        help="Solving method to use.",
        type=str.lower,
        choices=list(SOLVER_MAP.keys()),
        default=DEFAULTS['solver'],
    )
    parser.add_argument(
        "--update_optics",
        dest="update_optics",
        type=bool,
        help=("Sorts the RDTs to start with the highest order and updates the "
              "corrector strengths in the optics after calculation, so the "
              "feeddown to lower order correctors is included."
              ),
        default=DEFAULTS["update_optics"]
    )
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        help=("Reiterate correction, "
              "starting with the previously calculated values."),
        default=DEFAULTS["iterations"]
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


# Checks -----------------------------------------------------------------------

def check_opt(opt: Union[dict, DotDict]) -> DotDict:
    """ Asserts that the input parameters make sense and adds what's missing.
    If the input is empty, arguments will be parsed from commandline.
    Checks include:
        - Set defaults (see ``DEFAULTS``) if option not given.
        - Check accelerator name is valid
        - Set default RDTs if not given (see ``DEFAULT_RDTS``)
        - Check required parameters are present (twiss, errors, beams, rdts)
        - Check feeddown and iterations

    TODO: Replace DotDict with dataclass and have class check most of this...

    Args:
        opt (Union[dict, DotDict]): Function options in dictionary format.
                                Description of the arguments are given in
                                :func:`irnl_rdt_correction.main.irnl_rdt_correction`.

    Returns:
        DotDict: (Parsed and) checked options.

    """
    # check for unkown input
    parser = get_parser()
    if not len(opt):
        opt = vars(parser.parse_args())
    opt = DotDict(opt)
    known_opts = [a.dest for a in parser._actions if not isinstance(a, argparse._HelpAction)]  # best way I could figure out
    unknown_opts = [k for k in opt.keys() if k not in known_opts]
    if len(unknown_opts):
        raise AttributeError(f"Unknown arguments found: '{list2str(unknown_opts)}'.\n"
                             f"Allowed input parameters are: '{list2str(known_opts)}'")

    # Set defaults
    for name, default in DEFAULTS.items():
        if opt.get(name) is None:
            LOG.debug(f"Setting input '{name}' to default value '{default}'.")
            opt[name] = default

    # check accel
    opt.accel = opt.accel.lower()  # let's not care about case
    if opt.accel not in DEFAULT_RDTS.keys():
        raise ValueError(f"Parameter 'accel' needs to be one of '{list2str(list(DEFAULT_RDTS.keys()))}' "
                         f"but was '{opt.accel}' instead.")

    # Set rdts:
    if opt.get('rdts') is None:
        opt.rdts = DEFAULT_RDTS[opt.accel]

    # Check required and rdts:
    for name in ('twiss', 'errors', 'beams', 'rdts'):
        inputs = opt.get(name)
        if inputs is None or isinstance(inputs, str) or not isinstance(inputs, (Iterable, Sized)):
            raise ValueError(f"Parameter '{name}' is required and needs to be "
                             "iterable, even if only of length 1. "
                             f"Instead was '{inputs}'.")

    # Check twiss and errors input type
    for name in ('twiss', 'errors'):
        inputs = opt.get(name)
        for element in inputs:
            if not isinstance(element, (str, Path, pd.DataFrame, tfs.TfsDataFrame)):
                raise TypeError(f"Not all elements of '{name}' are DataFrames or paths to DataFrames!")

    if opt.feeddown < 0 or not (opt.feeddown == int(opt.feeddown)):
        raise ValueError("'feeddown' needs to be a positive integer.")

    if opt.iterations < 1:
        raise ValueError("At least one iteration (see: 'iterations') needs to "
                         "be done for correction.")
    return opt


