"""
PyDMD init
"""
__all__ = ['dmdbase', 'dmd', 'fbdmd', 'mrdmd', 'cdmd', 'hodmd', 'dmdc', 'optdmd']

__title__ = "pydmd"
__author__ = "Nicola Demo, Marco Tezzele"
__copyright__ = "Copyright 2017-2019, PyDMD contributors"
__license__ = "MIT"
__version__ = "0.3.1"
__mail__ = 'demo.nicola@gmail.com, marcotez@gmail.com'
__maintainer__ = __author__
__status__ = "Stable"


from .dmdbase import DMDBase
from .dmd import DMD
from .fbdmd import FbDMD
from .mrdmd import MrDMD
from .cdmd import CDMD
from .hodmd import HODMD
from .dmdc import DMDc
from .optdmd import OptDMD
