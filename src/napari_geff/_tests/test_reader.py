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


def test_reader_geff_no_time_axis(tmp_path, path_w_expected_graph_props):
    """
    Test that the reader returns None for a geff file with no time axis
    """
    # Use the fixture to create a valid file
    written_path, _ = path_w_expected_graph_props(
        np.uint16,
        {"position": "double"},
        {"score": np.float32, "color": np.uint8},
        directed=True,
    )

    # Open the zarr store to manipulate its attributes
    zarr_group = zarr.open(str(written_path), mode="a")
    original_axes = zarr_group.attrs["geff"]["axes"]

    # Create a new list of axes containing only the spatial ones
    spatial_axes_only = [
        axis for axis in original_axes if axis.get("type") != "time"
    ]
    zarr_group.attrs["geff"]["axes"] = spatial_axes_only

    # Write the modified attributes back to the .zattrs file
    with open(written_path / ".zattrs", "w") as f:
        json.dump(dict(zarr_group.attrs), f)

    # The reader should reject this file
    reader = get_geff_reader(str(written_path))
    assert reader is None, "Reader should be None for file without a time axis"


def test_reader_geff_no_space_axes(tmp_path, path_w_expected_graph_props):
    """
    Test that the reader returns None for a geff with no spatial axes
    """
    # Use the fixture to create a valid file
    written_path, _ = path_w_expected_graph_props(
        np.uint16,
        {"position": "double"},
        {"score": np.float32, "color": np.uint8},
        directed=True,
    )

    # Open the zarr store to manipulate its attributes
    zarr_group = zarr.open(str(written_path), mode="a")
    original_axes = zarr_group.attrs["geff"]["axes"]

    # Create a new list of axes containing only the time axis
    time_axis_only = [
        axis for axis in original_axes if axis.get("type") != "space"
    ]
    zarr_group.attrs["geff"]["axes"] = time_axis_only

    # Write the modified attributes back to the .zattrs file
    with open(written_path / ".zattrs", "w") as f:
        json.dump(dict(zarr_group.attrs), f)

    # The reader should reject this file
    reader = get_geff_reader(str(written_path))
    assert (
        reader is None
    ), "Reader should be None for file without spatial axes"


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
    assert all((True for _, item in edge_meta.items() if "score" in item))
    assert all((True for _, item in edge_meta.items() if "color" in item))
