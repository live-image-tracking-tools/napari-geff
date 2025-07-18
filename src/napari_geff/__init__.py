try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import get_geff_reader
from ._writer import write_multiple, write_single_image

__all__ = (
    "get_geff_reader",
    "write_single_image",
    "write_multiple",
)
