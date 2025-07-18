import json

import numpy as np
import pandas as pd
import zarr

from napari_geff._reader import get_geff_reader, reader_function


def test_reader_invalid_file(tmp_path):
    """Test that the reader returns None for an invalid file"""
    invalid_path = str(tmp_path / "invalid.zarr")
    reader = get_geff_reader(invalid_path)
    assert reader is None


def test_reader_valid_file(path_w_expected_graph_props):
    """Test valid file gets reader function"""
    written_path, props = path_w_expected_graph_props(
        np.uint16,
        {"position": "double"},
        {"score": np.float32, "color": np.uint8},
        directed=True,
    )
    reader = get_geff_reader(str(written_path))
    assert callable(reader), "Reader should be callable for valid geff file"


def test_reader_geff_no_axes(tmp_path, path_w_expected_graph_props):
    """Test that the reader returns None for a geff file without axes"""
    written_path, props = path_w_expected_graph_props(
        np.uint16,
        {"position": "double"},
        {"score": np.float32, "color": np.uint8},
        directed=True,
    )
    # read zattrs file and delete spatial axes attributes
    # then write it back out
    graph = zarr.open(str(written_path), mode="a")
    del graph.attrs["geff"]["axes"]
    with open(written_path / ".zattrs", "w") as f:
        json.dump(dict(graph.attrs), f)
    reader = get_geff_reader(str(written_path))
    assert (
        reader is None
    ), "Reader should return None for geff file without spatial data"


def test_reader_loads_layer(path_w_expected_graph_props):
    """Test the reader returns a tracks layer"""
    written_path, props = path_w_expected_graph_props(
        np.uint16,
        {"position": "double"},
        {"score": np.float32, "color": np.uint8},
        directed=True,
    )
    layer_tuples = reader_function(str(written_path))
    assert len(layer_tuples) == 1
    layer_tuple = layer_tuples[0]
    assert len(layer_tuple) == 3
    assert isinstance(layer_tuple[0], pd.DataFrame)
    assert isinstance(layer_tuple[1], dict)
    assert layer_tuple[2] == "tracks"


# TODO: update once test fixture writes out some node properties
def test_reader_loads_attrs(path_w_expected_graph_props):
    """Test the reader loads the expected attributes"""
    written_path, props = path_w_expected_graph_props(
        np.uint16,
        {"position": "double"},
        {"score": np.float32, "color": np.uint8},
        directed=True,
    )
    layer_tuples = reader_function(str(written_path))
    edge_meta = layer_tuples[0][1]["metadata"]["edge_properties"]
    assert "score" in edge_meta
    assert "color" in edge_meta
