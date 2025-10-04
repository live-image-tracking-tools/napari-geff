"""
This module provides a reader for geff zarr-backed files in napari.

If the file is a valid geff file with either position OR axis_names attributes,
the file will be read into a `Tracks` layer.

The original networkx graph read by `geff.read` is stored in the metadata
attribute on the layer.
"""

import contextlib
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

    zarrgeff = zarr.open(path, mode="r")
    np_to_pd_dtype = {
        np.dtype(np.int8): pd.Int8Dtype(),
        np.dtype(np.int16): pd.Int16Dtype(),
        np.dtype(np.int32): pd.Int32Dtype(),
        np.dtype(np.int64): pd.Int64Dtype(),
        np.dtype(np.uint8): pd.UInt8Dtype(),
        np.dtype(np.uint16): pd.UInt16Dtype(),
        np.dtype(np.uint32): pd.UInt32Dtype(),
        np.dtype(np.uint64): pd.UInt64Dtype(),
        np.dtype(np.float32): pd.Float32Dtype(),
        np.dtype(np.float64): pd.Float64Dtype(),
        np.dtype(np.bool): pd.BooleanDtype(),
    }
    for prop in zarrgeff["nodes"]["props"]:
        dtype = zarrgeff["nodes"]["props"][prop]["values"].dtype
        with contextlib.suppress(ValueError, TypeError):
            node_data_df[prop] = node_data_df[prop].astype(
                np_to_pd_dtype[dtype]
            )

    layers += [
        (
            tracks_napari,
            {
                "graph": track_graph,
                "name": "Tracks",
                "metadata": metadata,
                "features": node_data_df,
                "scale": scale,
                "translate": offset,
            },
            "tracks",
        )
    ]
    sort_order = ["image", "labels", "tracks"]
    layers = sorted(layers, key=lambda x: sort_order.index(x[2]))

    return layers
