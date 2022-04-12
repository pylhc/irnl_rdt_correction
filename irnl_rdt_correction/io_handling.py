"""
IO Handling
-----------

Functions for reading input and writing output.

"""
import logging
from pathlib import Path
from typing import Sequence, Iterable, Tuple

import tfs
from pandas import DataFrame
from tfs import TfsDataFrame

from irnl_rdt_correction.constants import PLANES, DELTA, EXT_MADX, EXT_TFS
from irnl_rdt_correction.utilities import is_anti_mirror_symmetric, idx2str, list2str

LOG = logging.getLogger(__name__)
X, Y = PLANES

# Input ------------------------------------------------------------------------

def get_tfs(paths: Sequence) -> Sequence[tfs.TfsDataFrame]:
    if isinstance(paths[0], str) or isinstance(paths[0], Path):
        return tuple(tfs.read_tfs(path, index="NAME") for path in paths)
    return paths


# Output -----------------------------------------------------------------------

def get_and_write_output(out_path: str, correctors: Sequence) -> Tuple[str, tfs.TfsDataFrame]:
    correction_text = build_correction_str(correctors)
    correction_df = build_correction_df(correctors)

    if out_path is not None:
        out_path = Path(out_path)
        write_command(out_path, correction_text)
        write_tfs(out_path, correction_df)

    return correction_text, correction_df


# Build ---

def build_correction_df(correctors: Sequence) -> TfsDataFrame:
    """ Build the DataFrame that contains the information for corrector powering."""
    attributes = vars(correctors[0])
    return TfsDataFrame(
        data=[[getattr(cor, attr) for attr in attributes] for cor in correctors],
        columns=attributes,
    )


def build_correction_str(correctors: Sequence) -> str:
    """ Build the MAD-X command that contains the information for corrector powering."""
    return _build_correction_str(corr for corr in correctors)


def build_correction_str_from_df(correctors_df: DataFrame) -> str:
    """ Build the MAD-X command that contains the information for corrector powering
    from the given DataFrame."""
    return _build_correction_str(row[1] for row in correctors_df.iterrows())


def _build_correction_str(iterator: Iterable) -> str:
    """ Creates madx commands (assignments) to run for correction"""
    last_type = ''
    text = ''
    for corr in iterator:
        if not _types_almost_equal(corr.type, last_type):
            text += f"\n!! {_nice_type(corr.type)} ({corr.field_component}) corrector\n"
            last_type = corr.type
        text += f"{corr.circuit} := {corr.value: 6E} / l.{corr.type} ;\n"
    return text.lstrip("\n")


def _types_almost_equal(type_a: str, type_b: str) -> bool:
    """ Groups correctors with and without ending F. """
    if type_a == type_b:
        return True
    if type_a.endswith('F'):
        return type_a[:-1] == type_b
    if type_b.endswith('F'):
        return type_b[:-1] == type_a


def _nice_type(mtype: str) -> str:
    """ Nicer naming with the optional 'F'"""
    if mtype.endswith('F'):
        return f'{mtype[:-1]}[F]'
    return mtype


# Write ---

def write_command(out_path: Path, correction_text: str):
    """ Write the correction string as textfile. """
    out_path_cmd = out_path.with_suffix(EXT_MADX)
    with open(out_path_cmd, "w") as out:
        out.write(correction_text)


def write_tfs(out_path: Path, correction_df: DataFrame):
    """ Write the correction dataframe as TFS-file. """
    out_path_df = out_path.with_suffix(EXT_TFS)
    tfs.write(out_path_df, correction_df)


# Utils ------------------------------------------------------------------------

def check_dfs(optics_dfs: Sequence[DataFrame], errors_dfs: Sequence[DataFrame],
              beams: Sequence[int], orders: Sequence[int], ignore_missing_columns: bool):
    """ Check the read optics and error dataframes for validity. """
    if len(optics_dfs) > 2 or len(errors_dfs) > 2:
        raise NotImplementedError("A maximum of two optics can be corrected "
                                  "at the same time, for now.")

    if len(optics_dfs) != len(errors_dfs):
        raise ValueError(f"The number of given optics ({len(optics_dfs):d}) "
                         "does not equal the number of given error files "
                         f"({len(errors_dfs):d}). Hint: it should.")

    if len(optics_dfs) != len(beams):
        raise ValueError(f"The number of given optics ({len(optics_dfs):d}) "
                         "does not equal the number of given beams "
                         f"({len(beams):d}). Please specify a beam for each "
                         "optics.")

    for idx_file, (optics, errors) in enumerate(zip(optics_dfs, errors_dfs)):
        not_found_errors = errors.index.difference(optics.index)
        if len(not_found_errors):
            raise IOError("The following elements were found in the "
                          f"{idx2str(idx_file)} given errors file but not in"
                          f"the optics: {list2str(not_found_errors.to_list())}")

        not_found_optics = optics.index.difference(errors.index)
        if len(not_found_optics):
            LOG.debug("The following elements were found in the "
                      f"{idx2str(idx_file)} given optics file but not in "
                      f"the errors: {list2str(not_found_optics.to_list())}."
                      f"They are assumed to be zero.")
            for indx in not_found_optics:
                errors.loc[indx, :] = 0

        needed_values = [f"K{order-1:d}{orientation}L"  # -1 for madx-order
                         for order in orders
                         for orientation in ("S", "")]

        for df, file_type in ((optics, "optics"), (errors, "error")):
            not_found_strengths = [s for s in needed_values if s not in df.columns]
            if len(not_found_strengths):
                text = ("Some strength values were not found in the "
                        f"{idx2str(idx_file)} given {file_type} file: "
                        f"{list2str(not_found_strengths)}.")

                if not ignore_missing_columns:
                    raise IOError(text)
                LOG.warning(text + " They are assumed to be zero.")
                for kl in not_found_strengths:
                    df[kl] = 0


def switch_signs_for_beams(optics_dfs: Iterable[tfs.TfsDataFrame],
                           error_dfs: Iterable[tfs.TfsDataFrame], beams: Iterable[int]):
    """ Switch the signs for Beam optics.
     This is due to the switch in direction between beam and
     (anti-) symmetry after a rotation of 180deg around the y-axis of magnets.
     Magnet orders that show anti-symmetry are: a1 (K0SL), b2 (K1L), a3 (K2SL), b4 (K3L) etc.

     This brings the Beam 2 KNL and x values to Beam 4 definition.
     """
    for idx, (optics_df, error_df, beam) in enumerate(zip(optics_dfs, error_dfs, beams)):
        if beam == 2:
            LOG.debug(f"Beam 2 input found. Switching signs for X and K(S)L values when needed.")
            optics_df[X] = -optics_df[X]
            error_df[f"{DELTA}{X}"] = -error_df[f"{DELTA}{X}"]

            # optics_df is already switched by MAD-X due to bv-flag.
            columns = error_df.columns[error_df.columns.map(is_anti_mirror_symmetric)]
            error_df[columns] = -error_df[columns]

            # Now Beam 2 optics and errors have the same values as Beam 4
            # i.e. the beam is now going "forward".
