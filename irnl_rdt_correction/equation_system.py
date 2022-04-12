"""
Equation System
---------------

Builds and solves the equation system from the rdt-to-corrector-maps given.

"""
import logging
from typing import Sequence, Tuple, Set

import numpy as np
from pandas import DataFrame, Series

from irnl_rdt_correction.constants import BETA, SIDES, PLANES, DELTA, KEYWORD, MULTIPOLE
from irnl_rdt_correction.rdt_handling import IRCorrector, RDT
from irnl_rdt_correction.utilities import (
    list2str, i_pow, is_even, is_odd, is_anti_mirror_symmetric, idx2str
)

LOG = logging.getLogger(__name__)

X, Y = PLANES


# Main Solving Function --------------------------------------------------------

def solve(rdt_maps, optics_dfs, errors_dfs, opt):
    """ Calculate corrections.
    They are grouped into rdt's with common correctors. If possible, these are
    ordered from highest order to lowest, to be able to update optics and include
    their feed-down. """
    all_correctors = []
    remaining_rdt_maps = rdt_maps
    while remaining_rdt_maps:
        current_rdt_maps, remaining_rdt_maps, corrector_names = get_current_rdt_maps(remaining_rdt_maps)

        for ip in opt.ips:
            correctors = get_available_correctors(corrector_names, opt.accel, ip, optics_dfs)
            all_correctors += correctors
            if not len(correctors):
                continue  # warnings are logged in get_available_correctors

            saved_corrector_values = init_corrector_and_optics_values(correctors, optics_dfs,
                                                                      opt.update_optics,
                                                                      opt.ignore_corrector_settings)

            beta_matrix, integral = build_equation_system(
                current_rdt_maps, correctors,
                ip, optics_dfs, errors_dfs, opt.beams, opt.feeddown,
            )
            for iteration in range(opt.iterations):
                solve_equation_system(correctors, beta_matrix, integral, opt.solver)  # changes corrector values
                update_optics(correctors, optics_dfs, opt.beams)  # update corrector values in optics

                # update integral values after iteration:
                integral_before = integral
                _, integral = build_equation_system(
                    current_rdt_maps, [],  # empty correctors list skips beta-matrix calculation
                    ip, optics_dfs, errors_dfs, opt.beams, opt.feeddown
                )
                _log_correction(integral_before, integral, current_rdt_maps, optics_dfs, iteration, ip)

            LOG.info(f"Correction of IP{ip:d} complete.")
            restore_optics_values(saved_corrector_values, optics_dfs)  # hint: nothing saved if update_optics is True
    return sorted(all_correctors)


# Preparation ------------------------------------------------------------------

def get_current_rdt_maps(rdt_maps):
    """ Creates a new rdt_map with all rdt's that share correctors.  """
    n_maps = len(rdt_maps)
    new_rdt_map = [{} for _ in rdt_maps]
    for rdt_map in rdt_maps:
        # get next RDT/correctors
        if len(rdt_map):
            rdt, correctors = list(rdt_map.items())[0]
            break
    else:
        raise ValueError("rdt_maps are empty. "
                         "This should have triggered an end of the solver loop "
                         "earlier. Please investigate.")

    correctors = set(correctors)
    checked_correctors = set()
    while len(correctors):
        # find all RDTs with the same correctors
        checked_correctors |= correctors
        additional_correctors = set()  # new correctors found this loop

        for corrector in correctors:
            for idx in range(n_maps):
                for rdt_current, rdt_correctors in rdt_maps[idx].items():
                    if corrector in rdt_correctors:
                        new_rdt_map[idx][rdt_current] = rdt_correctors
                        additional_correctors |= (set(rdt_correctors) - checked_correctors)

        correctors = additional_correctors

    remaining_rdt_map = [{k: v for k, v in rdt_maps[idx].items()
                          if k not in new_rdt_map[idx].keys()} for idx in range(n_maps)]

    if not any(len(rrm) for rrm in remaining_rdt_map):
        remaining_rdt_map = None

    return new_rdt_map, remaining_rdt_map, checked_correctors


def get_available_correctors(field_components: Set[str], accel: str, ip: int,
                             optics_dfs: Sequence[DataFrame]) -> Sequence[IRCorrector]:
    """ Gets the available correctors by checking for their presence in the optics.
    If the corrector is not found in this ip, the ip is skipped.
    If only one corrector (left or right) is present a warning is issued.
    If one corrector is present in only one optics (and not in the other)
    an Environment Error is raised. """
    correctors = []
    for field_component in field_components:
        corrector_not_found = []
        corrector_names = []
        for side in SIDES:
            corrector = IRCorrector(field_component, accel, ip, side)
            corr_in_optics = [_corrector_in_optics(corrector.name, df) for df in optics_dfs]
            if all(corr_in_optics):
                correctors.append(corrector)
                corrector_names.append(corrector.name)
            elif any(corr_in_optics):
                # Raise informative Error
                idx_true = corr_in_optics.index(True)
                idx_false = (idx_true + 1) % 2
                raise EnvironmentError(f'Corrector {corrector.name} was found'
                                       f'in the {idx2str(idx_true + 1)} optics'
                                       f'but not in the {idx2str(idx_false + 1)}'
                                       f'optics.')
            else:
                # Ignore Corrector
                corrector_not_found.append(corrector.name)

        if len(corrector_not_found) == 1:
            LOG.warning(f'Corrector {corrector_not_found[0]} could not be found in '
                        f'optics, yet {corrector_names[0]} was present. '
                        f'Correction will be performed with available corrector(s) only.')
        elif len(corrector_not_found):
            LOG.info(f'Correctors {list2str(corrector_not_found)} were not found in'
                     f' optics. Skipping IP{ip}.')

    return list(sorted(correctors))  # do not have to be sorted, but looks nicer


def init_corrector_and_optics_values(correctors: Sequence[IRCorrector], optics_dfs: Sequence[DataFrame],
                                     update_optics: bool, ignore_settings: bool):
    """ If wished save original corrector values from optics (for later restoration)
    and sync corrector values in list and optics. """
    saved_values = {}

    for corrector in correctors:
        values = [df.loc[corrector.name, corrector.strength_component] for df in optics_dfs]

        if not update_optics:
            saved_values[corrector] = values

        if ignore_settings:
            # set corrector value in optics to zero
            for df in optics_dfs:
                df.loc[corrector.name, corrector.strength_component] = 0
        else:
            if any(np.diff(values)):
                raise ValueError(f"Initial value for corrector {corrector.name} differs "
                                 f"between optics.")
            # use optics value as initial value
            corrector.value = values[0]
    return saved_values


# Build Equation System --------------------------------------------------------

def build_equation_system(rdt_maps: Sequence[dict], correctors: Sequence[IRCorrector], ip: int,
                          optics_dfs: Sequence, errors_dfs: Sequence, beams: Sequence,
                          feeddown: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Builds equation system as in Eq. (43) in [#DillyNonlinearIRCorrections2022]_
    for a given ip for all given optics and error files (i.e. beams) and rdts.

    Returns
        b_matrix: np.array N_rdts x  N_correctors
        integral: np.array N_rdts x 1
     """
    n_rdts = sum(len(rdt_map.keys()) for rdt_map, _ in zip(rdt_maps, optics_dfs))
    b_matrix = np.zeros([n_rdts, len(correctors)])
    integral = np.zeros([n_rdts, 1])

    idx_row = 0  # row in equation system
    for idx_file, rdts, optics_df, errors_df, beam in zip(range(1, 3), rdt_maps, optics_dfs, errors_dfs, beams):
        for rdt, rdt_correctors in rdts.items():
            LOG.info(f"Calculating {rdt}, optics {idx_file}/{len(optics_dfs)}, IP{ip:d}")
            integral[idx_row] = get_elements_integral(rdt, ip, optics_df, errors_df, feeddown)

            for idx_corrector, corrector in enumerate(correctors):
                if corrector.field_component not in rdt_correctors:
                    continue
                b_matrix[idx_row][idx_corrector] = get_corrector_coefficient(rdt, corrector, optics_df, errors_df, beam)
            idx_row += 1

    return b_matrix, integral


def get_elements_integral(rdt, ip, optics_df, errors_df, feeddown):
    """ Calculate the RDT integral for all elements of the IP. """
    integral = 0
    lm, jk = rdt.l + rdt.m, rdt.j + rdt.k
    # Integral on side ---
    for side in SIDES:
        LOG.debug(f" - Integral on side {side}.")
        side_sign = get_integral_sign(rdt.order, side)

        # get IP elements, errors and twiss have same elements because of check_dfs
        elements = optics_df.index[optics_df.index.str.match(fr".*{side}{ip:d}(\.B[12])?")]

        betax = optics_df.loc[elements, f"{BETA}{X}"]
        betay = optics_df.loc[elements, f"{BETA}{Y}"]
        if rdt.swap_beta_exp:
            # in case of beta-symmetry, this corrects for the same RDT in the opposite beam.
            betax = betax**(lm/2.)
            betay = betay**(jk/2.)
        else:
            betax = betax**(jk/2.)
            betay = betay**(lm/2.)

        dx = optics_df.loc[elements, X] + errors_df.loc[elements, f"{DELTA}{X}"]
        dy = optics_df.loc[elements, Y] + errors_df.loc[elements, f"{DELTA}{Y}"]
        dx_idy = dx + 1j*dy

        k_sum = Series(0j, index=elements)  # Complex sum of strengths (from K_n + iJ_n) and feed-down to them

        for q in range(feeddown+1):
            n_mad = rdt.order+q-1
            kl_opt = optics_df.loc[elements, f"K{n_mad:d}L"]
            kl_err = errors_df.loc[elements, f"K{n_mad:d}L"]
            iksl_opt = 1j*optics_df.loc[elements, f"K{n_mad:d}SL"]
            iksl_err = 1j*errors_df.loc[elements, f"K{n_mad:d}SL"]

            k_sum += ((kl_opt + kl_err + iksl_opt + iksl_err) *
                      (dx_idy**q) / np.math.factorial(q))

        integral += -sum(np.real(i_pow(lm) * k_sum.to_numpy()) * (side_sign * betax * betay).to_numpy())
    LOG.debug(f" -> Sum value: {integral}")
    return integral


def get_corrector_coefficient(rdt: RDT, corrector: IRCorrector, optics_df: DataFrame, errors_df: DataFrame, beam: int):
    """ Calculate B-Matrix Element for Corrector. """
    LOG.debug(f" - Corrector {corrector.name}.")
    lm, jk = rdt.l + rdt.m, rdt.j + rdt.k

    sign_i = np.real(i_pow(lm + (lm % 2)))  # i_pow is always real
    sign_corrector = sign_i * get_integral_sign(rdt.order, corrector.side)

    betax = optics_df.loc[corrector.name, f"{BETA}{X}"]
    betay = optics_df.loc[corrector.name, f"{BETA}{Y}"]
    if rdt.swap_beta_exp:
        # in case of beta-symmetry, this corrects for the same RDT in the opposite beam.
        betax = betax**(lm/2.)
        betay = betay**(jk/2.)
    else:
        betax = betax**(jk/2.)
        betay = betay**(lm/2.)

    z = 1
    p = corrector.order - rdt.order
    if p:
        # Corrector contributes via feed-down
        dx = optics_df.loc[corrector.name, X] + errors_df.loc[corrector.name, f"{DELTA}{X}"]
        dy = optics_df.loc[corrector.name, Y] + errors_df.loc[corrector.name, f"{DELTA}{Y}"]
        dx_idy = dx + 1j*dy
        z_cmplx = (dx_idy**p) / np.math.factorial(p)
        if (corrector.skew and is_odd(lm)) or (not corrector.skew and is_even(lm)):
            z = np.real(z_cmplx)
        else:
            z = np.imag(z_cmplx)
            if not corrector.skew:
                z = -z
        if abs(z) < 1e-15:
            LOG.warning(f"Z-coefficient for {corrector.name} in {rdt.name} is very small.")

    # Account for possible anti-symmetry of the correctors field component
    # for in Beam 2 and Beam 4. The correct sign in MAD-X is then assured by the
    # lattice setup, where these correctors have a minus sign in Beam 4.
    sign_beam = 1
    if is_even(beam) and is_anti_mirror_symmetric(corrector.strength_component):
        sign_beam = -1

    return sign_beam * sign_corrector * z * betax * betay


def get_integral_sign(n: int, side: str) -> int:
    """ Sign of the integral and corrector for this side.

    This is the exp(iπnθ(s_w−s_IP)) part of Eq. (40) in
    [#DillyNonlinearIRCorrections2022]_,
    """
    if side == "R":
        # return (-1)**n
        return -1 if n % 2 else 1
    return 1


# Solve ------------------------------------------------------------------------

def solve_equation_system(correctors: Sequence[IRCorrector], lhs: np.array, rhs: np.array, solver: str):
    """ Solves the system with the given solver.

    The result is transferred to the corrector values internally. """
    if len(rhs) > len(correctors) and solver not in APPROXIMATE_SOLVERS:
        raise ValueError("Overdetermined equation systems can only be solved "
                         "by one of the approximate solvers"
                         f" '{list2str(APPROXIMATE_SOLVERS)}'. "
                         f"Instead '{solver}' was chosen.")

    LOG.debug(f"Solving Equation system via {solver}.")
    # lhs x corrector = rhs <=> correctors = lhs\rhs
    # results are assigned to correctors directly
    SOLVER_MAP[solver](correctors, lhs, rhs)


# Solvers ---

def _solve_linear(correctors, lhs, rhs):
    """ Numpy solve. """
    res = np.linalg.solve(lhs, rhs)
    _assign_corrector_values(correctors, res)


def _solve_invert(correctors, lhs, rhs):
    """ Inverts the matrix. For test purposes only! """
    res = np.linalg.inv(lhs).dot(rhs)
    _assign_corrector_values(correctors, res)


def _solve_lstsq(correctors, lhs, rhs):
    """ Uses numpy's linear least squares. """
    res = np.linalg.lstsq(lhs, rhs, rcond=None)
    _assign_corrector_values(correctors, res[0])
    if len(res[1]):
        LOG.info(f"Residuals ||I - Bx||_2: {list2str(res[1])}")
    LOG.debug(f"Rank of Beta-Matrix: {res[2]}")


SOLVER_MAP = {'inv': _solve_invert,
              'lstsq': _solve_lstsq,
              'linear': _solve_linear,}


APPROXIMATE_SOLVERS = ['lstsq']


# Assign Results ---

def _assign_corrector_values(correctors: Sequence[IRCorrector], values: Sequence):
    """ Assigns calculated values to the correctors. """
    for corr, val in zip(correctors, values):
        if len(val) > 1:
            raise ValueError(f"Multiple Values for corrector {str(corr)} found."
                             f" There should be only one.")
        LOG.debug(f"Updating Corrector: {str(corr)} {val[0]:+.2e}.")
        corr.value += val[0]
        LOG.info(str(corr))


# Update Optics ----------------------------------------------------------------


def update_optics(correctors: Sequence[IRCorrector], optics_dfs: Sequence[DataFrame], beams: Sequence[int]):
    """ Updates the corrector strength values in the current optics. """
    for beam, optics in zip(beams, optics_dfs):
        for corrector in correctors:
            sign = -1 if is_even(beam) and is_anti_mirror_symmetric(corrector.strength_component) else 1
            optics.loc[corrector.name, corrector.strength_component] = sign * corrector.value


def restore_optics_values(saved_values: dict, optics_dfs: Sequence[DataFrame]):
    """ Restore saved initial corrector values (if any) to optics. """
    for corrector, values in saved_values.items():
        for df, val in zip(optics_dfs, values):
            df.loc[corrector.name, corrector.strength_component] = val


# Helper -----------------------------------------------------------------------

def _log_correction(integral_before: np.array, integral_after: np.array, rdt_maps: Sequence[dict],
                    optics_dfs: Sequence[DataFrame], iteration: int, ip: int):
    """ Log the correction initial and final value of this iteration. """
    LOG.info(f"RDT change in IP{ip:d}, iteration {iteration+1:d}:")
    integral_iter = zip(integral_before, integral_after)
    for idx_optics, rdts, _ in zip(range(1, 3), rdt_maps, optics_dfs):
        for rdt in rdts.keys():
            val_before, val_after = next(integral_iter)
            delta = val_after - val_before
            LOG.info(f"Optics {idx_optics}, {rdt.name}: {val_before[0]:.2e} -> {val_after[0]:.2e} ({delta[0]:+.2e})")


def _corrector_in_optics(name: str, df: DataFrame) -> bool:
    """ Checks if corrector is present in optics and has the KEYWORD MULTIPOLE. """
    if name not in df.index:
        LOG.debug(f'{name} not found in optics.')
        return False

    if KEYWORD in df.columns:
        if df.loc[name, KEYWORD].upper() != MULTIPOLE:
            LOG.warning(f"{name} found in optics, yet the Keyword was {df.loc[name, KEYWORD]}"
                        f" (should be '{MULTIPOLE}').")
            return False
    else:
        LOG.warning(f"'{KEYWORD}' column not found in optics."
                    f" Assumes you have filtered {MULTIPOLE}s beforehand!")

    return True




