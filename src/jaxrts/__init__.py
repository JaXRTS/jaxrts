# To generate the __init__.py automatically, use Erotemic's great
# mkinit: https://github.com/Erotemic/mkinit/
#
# Run the command 'mkinit --relative src/athens -w'

__version__ = "0.1.0"


# <AUTOGEN_INIT>
from . import bound_free
from . import core_electron
from . import free_bound
from . import helpers
from . import instrument_function
from . import ion_feature
from . import electron_feature
from . import plasmastate
from . import plotting
from . import units

__all__ = ['bound_free', 'core_electron', 'free_bound', 'helpers',
           'instrument_function', 'ion_feature', 'electron_feature', 'plasmastate', 'plotting']
# </AUTOGEN_INIT>
