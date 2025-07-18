import numpy as np
import zarr

from napari_geff import get_geff_reader


def test_reader_invalid_file(tmp_path):
    """Test that the reader returns None for an invalid file"""
    invalid_path = tmp_path / "invalid.zarr"
    reader = get_geff_reader(invalid_path)
    assert reader is None


def test_reader_valid_file(path_w_expected_graph_props):
    """Test valid file gets reader function"""
    written_path = path_w_expected_graph_props()
    reader = get_geff_reader(written_path)
    assert callable(reader), "Reader should be callable for valid geff file"


def test_reader_geff_no_axes(tmp_path, path_w_expected_graph_props):
    """Test that the reader returns None for a geff file without axes"""
    written_path, props = path_w_expected_graph_props(
        np.uint16,
        {"position": np.float32},
        {"score": np.float32, "color": np.uint8},
        directed=True,
    )
    # read zattrs file and delete pos and axis_names attributes
    # then write it back out
    graph = zarr.open(written_path, mode="a")
    del graph.attrs["axis_names"]
    del graph.attrs["position_prop"]
    del graph.attrs["axis_units"]
    del graph.attrs["roi_min"]
    del graph.attrs["roi_max"]
    reader = get_geff_reader(written_path)
    assert (
        reader is None
    ), "Reader should return None for geff file without spatial data"
