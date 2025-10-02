import json
from types import SimpleNamespace

import napari
import numpy as np
import pandas as pd
import zarr
from geff_spec import Axis, DisplayHint

from napari_geff._reader import (
    get_display_axes,
    get_geff_reader,
    reader_function,
)


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
    """Test the reader returns a tracks layer that is readable by the napari Viewer."""
    written_path, props = path_w_expected_graph_props(
        np.uint16,
        {"position": "double"},
        {"score": np.float32, "color": np.uint8},
        directed=True,
    )
    layer_tuples = reader_function(str(written_path))
    assert len(layer_tuples) == 1
    data, metadata, layer_type = layer_tuples[0]
    assert isinstance(data, pd.DataFrame)
    assert isinstance(metadata, dict)
    assert layer_type == "tracks"

    viewer = napari.Viewer()
    viewer.add_tracks(data, **metadata)
    viewer.close()


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


def test_display_axes_no_hints_all_provided():
    """Test that the display axes are correct when no display hints are provided."""
    # no display hints provided
    props = {}
    props["axes"] = [
        Axis(type="time", name="t"),
        Axis(type="space", name="z"),
        Axis(type="space", name="y"),
        Axis(type="space", name="x"),
    ]
    props["display_hints"] = None
    props_dot = SimpleNamespace(**props)
    display_axes, time_axis = get_display_axes(props_dot)
    assert display_axes == ["t", "z", "y", "x"]
    assert time_axis == "t"


def test_display_axes_no_hints_additional_spatial_axes_respects_time():
    """Test that display axes are correct with no display hints, time available, and additional spatial axes."""
    props = {}
    # no display hints and more than 4 spatiotemporal axes
    # innermost spatial axes returned, but time is always included
    props["axes"] = [
        Axis(type="time", name="t"),
        Axis(type="space", name="z"),
        Axis(type="space", name="c"),
        Axis(type="space", name="y"),
        Axis(type="space", name="x"),
    ]
    props["display_hints"] = None
    props_dot = SimpleNamespace(**props)
    display_axes, time_axis = get_display_axes(props_dot)
    assert display_axes == ["t", "c", "y", "x"]
    assert time_axis == "t"


def test_display_axes_no_hints_no_time():
    """Test that display axes are correct with no display hints and no time axis."""
    props = {}
    props["axes"] = [
        Axis(type="space", name="other"),
        Axis(type="space", name="z"),
        Axis(type="space", name="c"),
        Axis(type="space", name="y"),
        Axis(type="space", name="x"),
    ]
    props["display_hints"] = None
    props_dot = SimpleNamespace(**props)
    display_axes, time_axis = get_display_axes(props_dot)
    assert display_axes == ["z", "c", "y", "x"]
    assert time_axis is None


def test_display_axes_full_hints():
    """Test that display axes are correctly ordered with full display hints provided."""
    props = {}
    props["display_hints"] = DisplayHint(
        display_vertical="z", display_horizontal="y", display_depth="x"
    )
    props["axes"] = [
        Axis(type="time", name="t"),
        Axis(type="space", name="z"),
        Axis(type="space", name="y"),
        Axis(type="space", name="x"),
    ]
    props_dot = SimpleNamespace(**props)
    display_axes, time_axis = get_display_axes(props_dot)
    assert display_axes == ["t", "x", "z", "y"]


def test_display_axes_partial_hints():
    """Test that display axes are correctly ordered with partial display hints provided."""
    # fewer display hints than space axes, given axis order is respected
    props = {}
    props["display_hints"] = DisplayHint(
        display_vertical="z", display_horizontal="y"
    )
    props["axes"] = [
        Axis(type="time", name="t"),
        Axis(type="space", name="z"),
        Axis(type="space", name="y"),
        Axis(type="space", name="x"),
    ]
    props_dot = SimpleNamespace(**props)
    display_axes, time_axis = get_display_axes(props_dot)
    assert display_axes == ["t", "x", "z", "y"]
