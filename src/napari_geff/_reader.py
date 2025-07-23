"""
This module provides a reader for geff zarr-backed files in napari.

If the file is a valid geff file with either position OR axis_names attributes,
the file will be read into a `Tracks` layer.

The original networkx graph read by `geff.read_nx` is stored in the metadata
attribute on the layer.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

import geff
import pandas as pd
import pydantic
import zarr
from geff import GeffMetadata
from geff.utils import validate

from napari_geff.utils import get_tracklets_nx


def get_geff_reader(path: Union[str, list[str]]) -> Callable | None:
    """Returns reader function if path is a valid geff file, otherwise None.

    This function checks if the provided path is a valid geff file using the
    geff validator. It additionally checks that either a `position` or `axis_names`
    attribute is present on the graph, and that the graph is directed.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        Returns the reader function if the path is a valid geff file,
        otherwise returns None.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    try:
        validate(path)
    except (AssertionError, pydantic.ValidationError, ValueError):
        return None

    graph = zarr.open(path, mode="r")

    # graph attrs validation
    # Raises pydantic.ValidationError or ValueError
    meta = GeffMetadata(**graph.attrs["geff"])
    if meta.axes is None:
        return None
    has_time_axis = any(axis.type == "time" for axis in meta.axes)
    if not has_time_axis:
        return None  # Reject if no time axis is found, because tracks layers require time
    has_spatial_axes = any(axis.type == "space" for axis in meta.axes)
    if not has_spatial_axes:
        return None  # One also needs a spatial axis for napari tracks
    if not meta.directed:
        return None

    return reader_function


def reader_function(
    path: Union[str, list[str]],
) -> list[tuple[pd.DataFrame, dict[str, Any], str]]:
    """Read geff file at path and return `Tracks` layer data tuple.

    The original networkx graph read by `geff.read_nx` is stored in the metadata
    attribute on the layer.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tracks layer tuple
        List containing tuple of data and metadata for the `Tracks` layer
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    path = paths[0]

    nx_graph, geff_metadata = geff.read_nx(path, validate=False)

    layers = []
    if hasattr(geff_metadata, "related_objects"):
        related_objects = geff_metadata.related_objects
        if related_objects:
            for related_object in related_objects:
                if related_object.type == "labels":
                    labels_path = Path(path) / related_object.path
                    labels_path = os.path.expanduser(labels_path)
                    labels = zarr.open(labels_path, mode="r")
                    layers.append(
                        (
                            labels,
                            {
                                "name": "Labels",
                            },
                            "labels",
                        )
                    )
                if related_object.type == "image":
                    image_path = Path(path) / related_object.path
                    image_path = os.path.expanduser(image_path)
                    image = zarr.open(image_path, mode="r")
                    layers.append(
                        (
                            image,
                            {
                                "name": "Image",
                            },
                            "image",
                        )
                    )

    node_to_tid, track_graph = get_tracklets_nx(nx_graph)

    node_data_df = pd.DataFrame(nx_graph.nodes(data=True))
    node_data_df.rename(columns={0: "node_id"}, inplace=True)

    # Expand the 'props' column into multiple columns, don't use apply(pd.Series) on each row, since dtype won't be preserved
    expanded_cols_df = pd.DataFrame(
        node_data_df[1].tolist(), index=node_data_df.index
    )

    # Drop tbe original column of property dicts, and concat with the node_id
    node_data_df = pd.concat(
        [node_data_df.drop(columns=[1]), expanded_cols_df],
        axis=1,
    )
    node_data_df["napari_track_id"] = node_data_df["node_id"].map(node_to_tid)

    display_axes, time_axis_name = get_display_axes(geff_metadata)

    tracks_napari = node_data_df[(["napari_track_id"] + display_axes)]
    tracks_napari.sort_values(
        by=["napari_track_id", time_axis_name], inplace=True
    )  # Just in case

    metadata = {
        "nx_graph": nx_graph,
        "edge_properties": {
            (u, v): data for u, v, data in nx_graph.edges(data=True)
        },
        "geff_metadata": geff_metadata,
    }

    layers += [
        (
            tracks_napari,
            {
                "graph": track_graph,
                "name": "Tracks",
                "metadata": metadata,
                "features": node_data_df,
            },
            "tracks",
        )
    ]
    sort_order = ["image", "labels", "tracks"]
    layers = sorted(layers, key=lambda x: sort_order.index(x[2]))

    return layers


def get_display_axes(
    geff_metadata: GeffMetadata,
) -> tuple[list[str], str | None]:
    """Get display axes from geff metadata.

    Inspects geff_metadata.axes and geff_metadata.display_hints
    to determine the display axes in the order of time, depth, vertical,
    horizontal. At most 4 spatiotemporal axes are returned, even if
    more are present, as napari tracks layer only supports 4 axes on
    top of track ID.

    Parameters
    ----------
    geff_metadata : GeffMetadata
        Metadata object containing axis information.

    Returns
    -------
    list[str]
        List of display axes names in the order of time, depth, vertical, horizontal.
    """
    axes = geff_metadata.axes
    time_axis_name = None
    spatial_axes_names = []
    for axis in axes:
        if axis.type == "time":
            time_axis_name = axis.name
        elif axis.type == "space":
            spatial_axes_names.append(axis.name)

    # if display hints are provided, we make sure our spatial axis names
    # are ordered accordingly
    display_axis_dict = {}
    if geff_metadata.display_hints:
        display_hints = geff_metadata.display_hints
        if display_hints.display_depth:
            display_axis_dict["depth"] = display_hints.display_depth
        if display_hints.display_vertical:
            display_axis_dict["vertical"] = display_hints.display_vertical
        if display_hints.display_horizontal:
            display_axis_dict["horizontal"] = display_hints.display_horizontal
    display_axes = []
    for axis_type in ["depth", "vertical", "horizontal"]:
        if axis_type in display_axis_dict:
            display_axes.append(display_axis_dict[axis_type])
            spatial_axes_names.remove(display_axis_dict[axis_type])
    display_axes = spatial_axes_names + display_axes
    # we always take the time axis if we have it
    if time_axis_name:
        display_axes.insert(0, time_axis_name)
    if len(display_axes) > 4:
        # if there are more than 4 axes, we only take the innermost spatial axes
        # but we always include the time axis
        display_axes = (
            display_axes[-4:]
            if not time_axis_name
            else [display_axes[0]] + display_axes[-3:]
        )
    return display_axes, time_axis_name
