"""
This module provides a reader for geff zarr-backed files in napari.

If the file is a valid geff file with either position OR axis_names attributes,
the file will be read into a `Tracks` layer.

The original networkx graph read by `geff.read` is stored in the metadata
attribute on the layer.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

import geff
import numpy as np
import pandas as pd
import pydantic
import zarr
from geff import GeffMetadata

from napari_geff._features import build_typed_node_features
from napari_geff.utils import get_display_axes, get_tracklets_nx


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
        geff.validate_structure(path)
    except (
        AssertionError,
        pydantic.ValidationError,
        ValueError,
        FileNotFoundError,
    ):
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

    The original networkx graph read by `geff.read` is stored in the metadata
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

    nx_graph, geff_metadata = geff.read(path)

    scale = [ax.scale for ax in geff_metadata.axes]
    if not np.all([s is None for s in scale]):
        scale = [1 if s is None else s for s in scale]
    else:
        scale = None

    offset = [ax.offset for ax in geff_metadata.axes]
    if not np.all([o is None for o in offset]):
        offset = [0 if o is None else o for o in offset]
    else:
        offset = None

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
                                "scale": scale,
                                "translate": offset,
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
                                "scale": scale,
                                "translate": offset,
                            },
                            "image",
                        )
                    )

    node_to_tid, track_graph = get_tracklets_nx(nx_graph)

    # Determine the node order from the NetworkX graph (used for tracks rows)
    node_ids_in_nx_order = [n for n, _ in nx_graph.nodes(data=True)]

    # Build typed features directly from GEFF store, aligned to the NX node order
    typed_features_df = build_typed_node_features(
        path, geff_metadata, node_ids_in_nx_order
    )
    typed_features_df["napari_track_id"] = typed_features_df["node_id"].map(
        node_to_tid
    )

    display_axes, time_axis_name = get_display_axes(geff_metadata)

    tracks_napari = typed_features_df[["napari_track_id"] + display_axes]
    # Sort without inplace to avoid those pandas warnings
    tracks_napari = tracks_napari.sort_values(
        by=["napari_track_id", time_axis_name]
    )
    sorted_index = tracks_napari.index
    # Align features to the same sorted order and drop old index
    features_sorted = typed_features_df.loc[sorted_index].reset_index(
        drop=True
    )
    tracks_napari = tracks_napari.reset_index(drop=True)
    # Ensure napari "data" is a pure numeric NumPy array
    tracks_data_array = tracks_napari.astype(float).to_numpy()

    metadata = {
        "nx_graph": nx_graph,
        "edge_properties": {
            (u, v): data for u, v, data in nx_graph.edges(data=True)
        },
        "geff_metadata": geff_metadata,
    }

    layers += [
        (
            tracks_data_array,
            {
                "graph": track_graph,
                "name": "Tracks",
                "metadata": metadata,
                "features": features_sorted,
                "scale": scale,
                "translate": offset,
            },
            "tracks",
        )
    ]
    sort_order = ["image", "labels", "tracks"]
    layers = sorted(layers, key=lambda x: sort_order.index(x[2]))

    return layers
