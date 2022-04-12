"""
RDT Handling
------------

Class definitions of RDTs and IRCorrectors and functions handling these classes
for checking and sorting.

"""

import logging
from typing import Sequence, Tuple, Dict, List

from irnl_rdt_correction.constants import ORDER_NAME_MAP, SKEW_NAME_MAP, POSITION, SIDES
from irnl_rdt_correction.utilities import field_component2order, order2field_component, list2str

LOG = logging.getLogger(__name__)


# Classes ----------------------------------------------------------------------

class IRCorrector:
    """ Representation of an IR-Corrector in the accelerator. """
    def __init__(self, field_component: str, accel: str, ip: int, side: str):
        order, skew = field_component2order(field_component)

        self.field_component = field_component
        self.order = order
        self.skew = skew
        self.accel = accel
        self.ip = ip
        self.side = side

        main_name = f'C{ORDER_NAME_MAP[order]}{SKEW_NAME_MAP[skew]}X'
        extra = "F" if accel == "hllhc" and ip in [1, 5] else ""
        self.type = f'M{main_name}{extra}'
        self.name = f'{self.type}.{POSITION:d}{side}{ip:d}'
        self.circuit = f'K{main_name}{POSITION:d}.{side}{ip:d}'
        self.strength_component = f"K{order-1}{SKEW_NAME_MAP[skew]}L"  # MAD-X order notation
        self.value = 0

    def __repr__(self):
        return f"IRCorrector object {str(self)}"

    def __str__(self):
        return f"{self.name} ({self.field_component}), {self.strength_component}: {self.value: 6E}"

    def __lt__(self, other):
        if self.order == other.order:
            if self.skew == other.skew:
                if self.ip == other.ip:
                    return (self.side == SIDES[1]) < (other.side == SIDES[1])
                return self.ip < other.ip
            return self.skew < other.skew
        return self.order < other.order

    def __gt__(self, other):
        if self.order == other.order:
            if self.skew == other.skew:
                if self.ip == other.ip:
                    return (self.side == SIDES[1]) > (other.side == SIDES[1])
                return self.ip > other.ip
            return self.skew > other.skew
        return self.order > other.order

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class RDT:
    """ Representation of Resonance Driving Term attributes. """
    def __init__(self, name: str):
        self.name = name
        self.jklm = tuple(int(i) for i in name[1:5])
        self.j, self.k, self.l, self.m = self.jklm
        self.skew = bool((self.l + self.m) % 2)
        self.order = sum(self.jklm)
        self.swap_beta_exp = name.endswith("*")  # swap beta-exponents

    def __repr__(self):
        return f"RDT object {str(self)}"

    def __str__(self):
        return f"{self.name} ({order2field_component(self.order, self.skew)})"

    def __lt__(self, other):
        if self.order == other.order:
            if self.skew == other.skew:
                return len(self.name) < len(other.name)
            return self.skew < other.skew
        return self.order < other.order

    def __gt__(self, other):
        if self.order == other.order:
            if self.skew == other.skew:
                return len(self.name) > len(other.name)
            return self.skew > other.skew
        return self.order > other.order

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


# RDT Sorting ------------------------------------------------------------------

def sort_rdts(rdts: Sequence, rdts2: Sequence) -> Tuple[dict, dict]:
    """ Sorts RDTs by reversed-order and orientation (skew, normal). """
    LOG.debug("Sorting RDTs")
    LOG.debug(" - First Optics:")
    rdt_dict = _build_rdt_dict(rdts)
    if rdts2 is not None:
        LOG.debug(" - Second Optics:")
        rdt_dict2 = _build_rdt_dict(rdts2)
    else:
        LOG.debug(" - Second Optics: same RDTs as first.")
        rdt_dict2 = rdt_dict.copy()
    return rdt_dict, rdt_dict2


def _build_rdt_dict(rdts: Sequence) -> Dict[RDT, List[str]]:
    LOG.debug("Building RDT dictionary.")
    if not isinstance(rdts, dict):
        rdts = {rdt: [] for rdt in rdts}

    rdt_dict = {}
    for rdt_name, correctors in rdts.items():
        rdt = RDT(rdt_name)
        if not len(correctors):
            skew = rdt.skew
            correctors = [order2field_component(rdt.order, skew)]

        rdt_dict[rdt] = correctors
        LOG.debug(f"Added: {rdt} with correctors: {list2str(correctors)}")
    rdt_dict = dict(sorted(rdt_dict.items(), reverse=True))  # sorts by highest order and skew first
    return rdt_dict


def get_needed_orders(rdt_maps: Sequence[dict], feed_down: int) -> Sequence[int]:
    """Returns the sorted orders needed for correction, based on the order
    of the RDTs to correct plus the feed-down involved and the order of the
    corrector, which can be higher than the RDTs in case one wants to correct
    via feed-down."""
    needed_orders = set()
    for rdt_map in rdt_maps:
        for rdt, correctors in rdt_map.items():
            # get orders from RDTs + feed-down
            for fd in range(feed_down+1):
                needed_orders |= {rdt.order + fd, }

            # get orders from correctors
            for corrector in correctors:
                needed_orders |= {int(corrector[1]), }
    return sorted(needed_orders)


# Order Checks ----
def check_corrector_order(rdt_maps: Sequence[dict], update_optics: bool, feed_down: int):
    """ Perform checks on corrector orders compared to RDT orders and feed-down. """
    for rdt_map in rdt_maps:
        for rdt, correctors in rdt_map.items():
            _check_corrector_order_not_lower(rdt, correctors)
            _check_update_optics(rdt, correctors, rdt_map, update_optics, feed_down)


def _check_update_optics(rdt: RDT, correctors: list, rdt_map: dict, update_optics: bool, feed_down: int):
    """ Check if corrector values are actually set before they are needed for feed-down.
    Otherwise an error is thrown. This is only problematic if `update_optics` is `True`.

    The problem arises, if a corrector is defined via the calculation of
    feed-down to a lower-order rdt, but there are other rdts that also
    depend on (the feeddown or the value) of this corrector, which have
    higher order than the rdt that defines the corrector value and hence
    would be calculated earlier.

    E.g. b6 corrector value is calculated from correcting octupole RDT via
    feed-down. Yet decapole RDT also needs correction and the feed-down value
    is >= 1.

    It should be possible to mitigate this by sorting the RDTs by their
    highest corrector order instead of their own order.
    (one could make a new field `sort_order`. To be tested.)
    """
    if not update_optics:
        return

    for corrector in correctors:
        corrector_order = int(corrector[1])
        for rdt_comp in rdt_map.keys():  # compare with all rdts
            if rdt_comp is rdt:
                # rdts are sorted high -> low
                # so from here on everything is fine as all lower rdts are
                # calculated later anyway.
                break
            if rdt_comp.order < corrector_order <= rdt_comp.order + feed_down:
                # as we did not break yet, rdt_comp.order must be higher or
                # equal to rdt.order. Yet with the current feed-down value
                # the corrector's strength is needed for rdt_comp.
                raise ValueError(
                    "Updating the optics is in this configuration not possible,"
                    f" as corrector {corrector} influences {rdt_comp.name}"
                    f" with the given feed-down of {feed_down}. Yet the value of"
                    f" the corrector is defined by {rdt.name}.")


def _check_corrector_order_not_lower(rdt: RDT, correctors: List[str]):
    """ Check if only higher and equal order correctors are defined to correct
    a given rdt."""
    wrong_correctors = [c for c in correctors if int(c[1]) < rdt.order]
    if len(wrong_correctors):
        raise ValueError(
            "Correctors can not correct RDTs of higher order."
            f" Yet for {rdt.name} the corrector(s)"
            f" '{list2str(wrong_correctors)}' was (were) given."
        )
