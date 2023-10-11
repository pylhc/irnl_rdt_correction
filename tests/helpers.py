from contextlib import contextmanager
import re
from pathlib import Path
import sys
from typing import Union, List, Iterable, Sequence

import h5py
import pandas as pd
from tfs import TfsDataFrame

from irnl_rdt_correction.constants import BETA, KEYWORD, PLANES, MULTIPOLE

X, Y = PLANES

INPUTS = Path(__file__).parent / "inputs"
LHC_MODELS_PATH = INPUTS / "model_lhc_thin_30cm"

EPS = 1e-13  # to compare floating point numbers against
ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # the alphabet
MAX_N = 6  # 2 == Sextupole

PLACEHOLDER = "PLACEHOLDER"  # MADX Keyword PLACEHOLDER

# fields of IRCorrector --> columns in corrections tfs
VALUE = "value"
STRENGTH = "strength_component"
FIELD = "field_component"
ORDER = "order"
IP = "ip"
CIRCUIT = "circuit"
NAME = "name"


def read_hdf(path: Union[Path, str]) -> TfsDataFrame:
    """Read TfsDataFrame from hdf5 file. The DataFrame needs to be stored
    in a group named ``data``, while the headers are stored in ``headers``.

    Args:
        path (Path, str): Path of the file to read.
    """
    df = pd.read_hdf(path, key="data")
    with h5py.File(path, mode="r") as hf:
        headers = hf.get('headers')
        headers = {k: headers[k][()] for k in headers.keys()}

    for key, value in headers.items():
        try:
            headers[key] = value.decode('utf-8')  # converts byte-strings back
        except AttributeError:
            pass  # probably numeric
    return TfsDataFrame(df, headers=headers)


def read_lhc_model(beam: int) -> TfsDataFrame:
    """Read the LHC model from the input directory."""
    # tfs files were too big, but if generated from the `.madx` the `.tfs` can be used directly.
    # E.g. for debugging purposes.
    # return tfs.read_tfs(LHC_MODELS_PATH / f"twiss.lhc.b{beam}.nominal.tfs", index="NAME")
    return read_hdf(LHC_MODELS_PATH / f"twiss.lhc.b{beam}.nominal.hd5")


def generate_pseudo_model(n_ips: int, n_magnets: int, accel: str,
                          betax: float = 1, betay: float = 1, x: float = 0, y: float = 0) -> pd.DataFrame:
    """Generate a Twiss-Like DataFrame with magnets as index and Beta and Orbit columns."""
    df = pd.DataFrame(
        index=(
                get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets) +
                get_lhc_corrector_names(n_ips=n_ips, accelerator=accel)
        ),
        columns=[f"{BETA}{X}", f"{BETA}{Y}", X, Y, KEYWORD]
    )
    df[f"{BETA}{X}"] = betax
    df[f"{BETA}{Y}"] = betay
    df[X] = x
    df[Y] = y
    df[KEYWORD] = MULTIPOLE
    return df


def generate_errortable(index: Sequence, value: float = 0, max_order: int = MAX_N) -> pd.DataFrame:
    """Return DataFrame from index and KN(S)L + D[XY] columns."""
    return pd.DataFrame(value,
                        index=index,
                        columns=[f"K{n}{o}L" for n in range(max_order) for o in ("", "S")] + [f"D{plane}" for plane in "XY"]
                        )


def get_some_magnet_names(n_ips: int, n_magnets: int, sides: Iterable = "LR") -> List[str]:
    r"""More or less random magnet names, ending in ``[LR]\d``.
    n_magnets < 26 because their names come from alphabet.
    """
    return [
        f"M{name}.{number+1}{side}{ip}"
        for ip in range(1, n_ips+1)
        for side in sides
        for number, name in enumerate(ABC[:n_magnets])
    ]


def get_lhc_corrector_names(n_ips: int, accelerator: str = 'lhc') -> List[str]:
    r"""Corrector names as defined in LHC/HLLHC as the correction script is looking for them.
    Need to start with ``MC`` and end in ``X.3[LR]\d`` or ``XF.3[LR][15]``"""
    magnets = [
        f"MC{order}{orientation}X.3{side}{ip}"
        for order in "SODT"
        for orientation in ("S", "")
        for side in "LR"
        for ip in range(1, n_ips+1)
    ]
    if accelerator == 'hllhc':
        magnets = [
            name.replace("X", "XF") if name[-1] in "15" else name
            for name in magnets
        ]
    return magnets


def get_ir_magnets_mask(index: pd.Index) -> pd.Series:
    """Returns a boolean mask for magnets in the IR (n<=13) in the index."""
    return index.str.match(r"M.*\.(1[0123]|[0-9])[LR]\d(\.B\d)?", flags=re.IGNORECASE)


def get_corrector_magnets_mask(index: pd.Index) -> pd.Series:
    """Returns a boolean mask for the nonlinear corrector magnets in index."""
    return index.str.match(r"MC.*XF?\.3[LR]\d$", flags=re.IGNORECASE)


def get_opposite_sign_beam4_kl_columns(range_: Iterable):
    """Get the KN(S)L columns that have opposite signs for beam 4."""
    return [f"K{order}{'' if order % 2 else 'S'}L" for order in range_]


@contextmanager
def cli_args(*args, **kwargs):
    """ Provides context to run an entrypoint like with commandline args.
    Arguments are restored after context.

    Args:
        The Commandline args (excluding the script name)

    Keyword Args:
        script: script-name. Used as first commandline-arg.
                Otherwise it's 'somescript.py'
    """
    script = kwargs.get("script", "somescript.py")
    args_save = sys.argv.copy()
    sys.argv = [script] + list(args)
    yield
    sys.argv = args_save
