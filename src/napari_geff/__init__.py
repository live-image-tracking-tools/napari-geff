try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import get_geff_reader

__all__ = ("get_geff_reader",)
