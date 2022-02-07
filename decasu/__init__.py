try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("decasu")
except PackageNotFoundError:
    # package is not installed
    pass

from .configuration import Configuration
from .simple_healpix_mapper import SimpleHealpixMapper
from .multi_healpix_mapper import MultiHealpixMapper
from .wcs_table import WcsTableBuilder
from .multi_tile_mapper import MultiTileMapper
from .region_mapper import RegionMapper

