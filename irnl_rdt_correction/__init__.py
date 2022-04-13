"""
IRNL RDT Correction
~~~~~~~~~~~~~~~~~~~

Correction script to power the nonlinear correctors in the (HL-)LHC insertion regions based on RDTs.

:copyright: pyLHC/OMC-Team working group.
:license: MIT, see the LICENSE.md file for details.
"""
from irnl_rdt_correction.main import irnl_rdt_correction

__title__ = "irnl-rdt-correction"
__description__ = "Correction script to power the nonlinear correctors in the (HL-)LHC insertion regions based on RDTs."
__url__ = "https://github.com/pylhc/irnl_rdt_correction"
__version__ = "1.0.0"
__author__ = "pylhc"
__author_email__ = "pylhc@github.com"
__license__ = "MIT"

__all__ = [irnl_rdt_correction, __version__]
