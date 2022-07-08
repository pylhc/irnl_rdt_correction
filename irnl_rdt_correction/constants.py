"""
Constants
---------

These constants rely a lot on the HL-LHC/LHC Naming Scheme.
"""
from pathlib import Path
from typing import Union, Dict, Sequence

from pandas import DataFrame
from tfs import TfsDataFrame

# Mappings ---
ORDER_NAME_MAP = {1: "B", 2: "Q", 3: "S", 4: "O", 5: "D", 6: "T"}
SKEW_NAME_MAP = {True: "S", False: ""}
SKEW_FIELD_MAP = {True: "a", False: "b"}
FIELD_SKEW_MAP = {v: k for k, v in SKEW_FIELD_MAP.items()}
SKEW_CHAR_MAP = {True: "J", False: "K"}

# Corrector Names ---
POSITION = 3  # all NL correctors are at POSITION 3, to be adapted for Linear
SIDES = ("L", "R")

# Columns ---
PLANES = ("X", "Y")
BETA = "BET"
DELTA = "D"
KEYWORD = "KEYWORD"
MULTIPOLE = "MULTIPOLE"

# File Extensions ---
EXT_TFS = ".tfs"  # suffix for dataframe file
EXT_MADX = ".madx"  # suffix for madx-command file

# Types ---
StrOrPathOrDataFrame = Union[str, Path, DataFrame, TfsDataFrame]
RDTInputTypes = Union[Sequence[str], Dict[str, Sequence[str]]]
