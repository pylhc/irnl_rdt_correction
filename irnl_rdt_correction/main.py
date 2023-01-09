"""
Nonlinear Correction calculation in the IRs
--------------------------------------------

Performs local correction of the Resonance Driving Terms (RDTs)
in the Insertion Regions (IRs) based on the principle described in
[BruningDynamicApertureStudies2004]_ with the addition of correcting
feed-down and using feed-down to correct lower order RDTs.
Details can be found in [DillyNonlinearIRCorrections2023]_ .

TODO:
 - [easy] Allow not giving errors (need to be `None` in list,
   so that the list lengths are still the same and there is a
   clear correspondence twiss-errors-beams).
   Should then be assumed all zero.
 - [easy] Allow for more than two optics given
   (e.g. find corrections for 15cm and 30cm for both beams)
 - [medium] Maybe sort RDTs by highest corrector instead of highest RDT order?
   This should allow for correctors that correct via feed-down
   to be assigned before lower order RDTs are calculated.
   Might cause other problems, though. To be thought about.
 - [medium] Consider switching the signs all into the reference frame of Beam 1.
   That means X, DX and anti-mirror-KN(S)L twiss and errors from Beam 4,
   and the anti-mirror-KN(S)L twiss from Beam 2.
   That should in principle allow to ignore all other beam-related sign switches.
   BUT: does this really work with all the feed-down options implemented
   (i.e. feed-down to RDT, feed-down from correctors)?
   It should, but needs to be checked and tested.
 - [medium] Take phase advance between the elements and to the correction point 
   at the entrance of the IR into account. 
   That would mean correct the numerator of the actual RDT.
 - [hard] Additionally to taking the phase-advance into account, one might try to optimize
   the actual RDTs at the position of the correctors. This might be very problematic,
   as we have two correctors (one on each side) per order, so that might become a
   non-linear problem (as now there are now two equations, one per corrector, which
   are non-linearly dependent.)


.. rubric:: References

..  [BruningDynamicApertureStudies2004]
    O. Bruening et al.,
    Dynamic aperture studies for the LHC separation dipoles. (2004)
    https://cds.cern.ch/record/742967

..  [DillyNonlinearIRCorrections2023]
    J. Dilly et R. Tomás,
    A flexible nonlinear Resonance Driving Term based Correction Algorithm with feed-down. (2023)
    https://github.com/pylhc/irnl_rdt_correction/blob/master/latex/note.pdf


author: Joschua Dilly

"""
import logging
from typing import Tuple

import tfs

from irnl_rdt_correction.equation_system import solve
from irnl_rdt_correction.input_options import check_opt
from irnl_rdt_correction.io_handling import get_and_write_output, get_optics
from irnl_rdt_correction.rdt_handling import sort_rdts, check_corrector_order, get_needed_orders
from irnl_rdt_correction.utilities import Timer, log_setup

LOG = logging.getLogger(__name__)


def irnl_rdt_correction(**opt) -> Tuple[str, tfs.TfsDataFrame]:
    """ Get correctors and their optimal powering to minimize the given RDTs.

    Keyword Args:

        twiss (list[str/Path/DataFrame]): Path(s) to twiss file(s) or DataFrame(s) of optics.
                                           Needs to contain only the elements to be corrected
                                           (e.g. only the ones in the IRs).
                                           All elements from the error-files need to be present.
                                           Required!
        errors (list[str/Path/DataFrame]): Path(s) to error file(s) or DataFrame(s) of errors.
                                           Can contain less elements than the optics files,
                                           these elements are then assumed to have no errors.
                                           Required!
        beams (list[int]): Which beam the files come from (1, 2 or 4).
                           Required!
        output (str/Path): Path to write command and tfs_df file.
                           Extension (if given) is ignored and replaced with '.tfs' and '.madx'
                           for the Dataframe and the command file respectively.
                           Default: ``None``.
        rdts (list[str], dict[str, list[str]):
                          RDTs to correct.
                          Format: 'Fjklm'; or 'Fjklm*' to correct for
                          this RDT in Beam 2 using beta-symmetry (jk <-> lm).
                          The RDTs can be either given as a list, then the appropriate correctors
                          are determined by jklmn.
                          Alternatively, the input can be a dictionary,
                          where the keys are the RDT names as before, and the values are a list
                          of correctors to use, e.g. 'b5' for normal decapole corrector,
                          'a3' for skew sextupole, etc.
                          If the order of the corrector is higher than the order of the RDT,
                          the feed-down from the corrector is used for correction.
                          In the case where multiple orders of correctors are used,
                          increasing ``iterations`` might be useful.
                          Default: Standard RDTs for given ``accel`` (see ``DEFAULT_RDTS`` in this file).
        rdts2 (list[str], dict[str, list[str]):
                           RDTs to correct for second beam/file, if different from first.
                           Same format rules as for ``rdts``. Default: ``None``.
        accel (str): Which accelerator we have. One of 'lhc', 'hllhc'.
                     Default: ``lhc``.
        feeddown (int): Order of Feeddown to include.
                        Default: ``0``.
        ips (list[int]): In which IPs to correct.
                         Default: ``[1, 2, 5, 8]``.
        solver (str): Solver to use. One of 'lstsq', 'inv' or 'linear'.
                      Default ``lstsq``.
        update_optics (bool): Sorts the RDTs to start with the highest order
                              and updates the corrector strengths in the optics
                              after calculation, so the feeddown to lower order
                              correctors is included.
                              Default: ``True``.
        ignore_corrector_settings (bool): Ignore the current settings of the correctors.
                                          If this is not set the corrector values of the
                                          optics are used as initial conditions.
                                          Default: ``False``.
        ignore_missing_columns (bool): If set, missing strength columns in any
                                       of the input files are assumed to be
                                       zero, instead of raising an error.
                                       Default: ``False``.
        iterations (int): (Re-)iterate correction, starting with the previously
                          calculated values. Needs to be > 0, as the first calculation
                          counts as an iteration.
                          Default: ``1``.


    Returns:

        tuple[string, Dataframe]:
        the string contains the madx-commands used to power the correctors;
        the dataframe contains the same values in a pandas DataFrame.
    """
    LOG.info("Starting IRNL Correction.")
    timer = Timer("Start", print_fun=LOG.debug)
    opt = check_opt(opt)
    timer.step("Opt Parsed.")

    rdt_maps = sort_rdts(opt.rdts, opt.rdts2)
    check_corrector_order(rdt_maps, update_optics=opt.update_optics, feed_down=opt.feeddown)
    needed_orders = get_needed_orders(rdt_maps, opt.feeddown)
    timer.step("RDT Sorted.")

    optics = get_optics(
        beams=opt.beams, twiss=opt.twiss, errors=opt.errors,
        orders=needed_orders,
        ignore_missing_columns=opt.ignore_missing_columns
    )
    timer.step("Optics Loaded.")

    correctors = solve(
        rdt_maps=rdt_maps,
        optics_seq=optics,
        accel=opt.accel,
        ips=opt.ips,
        update_optics=opt.update_optics,
        ignore_corrector_settings=opt.ignore_corrector_settings,
        feeddown=opt.feeddown,
        iterations=opt.iterations,
        solver=opt.solver
    )
    timer.step("Done")

    timer.summary()
    if len(correctors) == 0:
        raise EnvironmentError('No correctors found in input optics.')

    return get_and_write_output(opt.output, correctors)


# Script Mode ------------------------------------------------------------------

if __name__ == '__main__':
    log_setup()
    irnl_rdt_correction()
