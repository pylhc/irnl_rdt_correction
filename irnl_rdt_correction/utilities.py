"""
Utilities
---------
Additional utilities used throughout the correction.

"""
import logging
import re
import sys
from dataclasses import dataclass
from time import time
from typing import Callable, Tuple

from pandas import DataFrame
from tfs import TfsDataFrame

from irnl_rdt_correction.constants import SKEW_NAME_MAP, SKEW_FIELD_MAP, FIELD_SKEW_MAP


# Classes ----------------------------------------------------------------------

class DotDict(dict):
    """ Make dict fields accessible by attributes.
    TODO: Replace with dataclass.
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        """ Needed to raise the correct exceptions """
        try:
            return super(DotDict, self).__getitem__(key)
        except KeyError as e:
            raise AttributeError(e).with_traceback(e.__traceback__) from e


class Timer:
    """ Collect Times and print a summary at the end. """
    def __init__(self, name: str = "start", print_fun: Callable[[str], None] = print):
        self.steps = {}
        self.step(name)
        self.print = print_fun

    def step(self, name: str = None) -> float:
        if not name:
            name = str(len(self.steps))
        time_ = time()
        self.steps[name] = time_
        return time_

    def time_since_step(self, step=None) -> float:
        if not step:
            step = list(self.steps.keys())[-1]
        dtime = time() - self.steps[step]
        return dtime

    def time_between_steps(self, start: str = None, end: str = None) -> float:
        list_steps = list(self.steps.keys())
        if not start:
            start = list_steps[0]
        if not end:
            end = list_steps[-1]
        dtime = self.steps[end] - self.steps[start]
        return dtime

    def summary(self):
        str_length = max(len(s) for s in self.steps.keys())
        time_length = len(f"{int(self.time_between_steps()):d}")
        format_str = (f"{{step:{str_length}s}}:"
                      f" +{{dtime: {time_length:d}.5f}}s"
                      f" ({{ttime: {time_length:d}.3f}}s total)")
        last_time = None
        start_time = None
        for step, step_time in self.steps.items():
            if last_time is None:
                start_time = step_time
                self.print(f"Timing Summary ----")
                self.print(format_str.format(step=step, dtime=0, ttime=0))
            else:
                self.print(format_str.format(
                    step=step, dtime=step_time-last_time, ttime=step_time-start_time)
                )
            last_time = step_time


@dataclass
class Optics:
    """ Store Optics Data. """
    beam: int
    twiss: DataFrame
    errors: DataFrame


# KNL-Checks -------------------------------------------------------------------

def get_max_knl_order(df: TfsDataFrame) -> int:
    """ Return the maximum order in the KN(S)L columns of the DataFrame. """
    return df.columns.str.extract(r"^K(\d+)S?L$", expand=False).dropna().astype(int).max()


def is_anti_mirror_symmetric(column_name: str) -> bool:
    """ Returns true if the column name is a KNL/KNSL column and the
    magnetic field that this column represents is anti-symmetric upon mirroring on y axis."""
    try:
        order = int(re.match(r"^K(\d+)S?L$", column_name, flags=re.IGNORECASE).group(1))
    except AttributeError:
        # name does not match pattern
        return False
    else:
        return column_name.upper() == f"K{order:d}{SKEW_NAME_MAP[is_even(order)]}L"


# Small Helpers ----------------------------------------------------------------

def list2str(list_: list) -> str:
    return str(list_).strip('[]')


def idx2str(idx: int) -> str:
    return {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}[idx+1]


def order2field_component(order: int, skew: bool) -> str:
    return f"{SKEW_FIELD_MAP[skew]:s}{order:d}"


def field_component2order(field_component) -> Tuple[int, bool]:
    return int(field_component[1]), FIELD_SKEW_MAP[field_component[0]]


def is_odd(n: int) -> bool:
    return bool(n % 2)


def is_even(n: int) -> bool:
    return not is_odd(n)


def i_pow(n: int) -> complex:
    """ i to the power of n."""
    return 1j**(n % 4)   # more exact with modulo


def log_setup():
    """ Set up a basic logger. """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(levelname)7s | %(message)s | %(name)s"
    )

