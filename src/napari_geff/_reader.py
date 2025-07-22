"""
This module provides a reader for geff zarr-backed files in napari.

If the file is a valid geff file with either position OR axis_names attributes,
the file will be read into a `Tracks` layer.

The original networkx graph read by `geff.read_nx` is stored in the metadata
attribute on the layer.
"""

import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

import geff
import networkx as nx
import numpy as np
import pandas as pd
import pydantic
import zarr
from geff import GeffMetadata
from geff.utils import validate


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
    except (AssertionError, pydantic.ValidationError):
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

    node_to_tid, track_graph = get_tracklets(nx_graph)

    node_data_df = pd.DataFrame(nx_graph.nodes(data=True))
    node_data_df.rename(columns={0: "node_id"}, inplace=True)

    # Expand the 'props' column into multiple columns, don't use apply(pd.Series) on each row, since dtype won't be preserved
    # TODO: they aren't preserved anyway if there's a nan in there
    expanded_cols_df = pd.DataFrame(
        node_data_df[1].tolist(), index=node_data_df.index
    )

    # Drop tbe original column of property dicts, and concat with the node_id
    node_data_df = pd.concat(
        [node_data_df.drop(columns=[1]), expanded_cols_df],
        axis=1,
    )

    node_data_df["napari_track_id"] = node_data_df["node_id"].map(node_to_tid)
    axes = geff_metadata.axes
    time_axis_name = None
    spatial_axes_names = []
    for axis in axes:
        if axis.type == "time":
            time_axis_name = axis.name
        elif axis.type == "space":
            spatial_axes_names.append(axis.name)

    tracks_napari = node_data_df[
        (
            ["napari_track_id"]
            + ([time_axis_name] if time_axis_name else [])
            + spatial_axes_names
        )
    ]
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
        if dtype in np_to_pd_dtype:
            node_data_df[prop] = node_data_df[prop].astype(
                np_to_pd_dtype[dtype]
            )
        else:
            pass

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

    return layers


def get_tracklets(
    graph: nx.DiGraph,
) -> tuple[dict[Any, int], dict[int, list[int]]]:
    """Extract tracklet IDs and parent-child connections from a directed graph.

    A tracklet consists of a sequence of nodes in the graph connected by edges
    where the incoming and outgoing degree of each node on the path is at most 1.

    Parameters
    ----------
    graph : nx.DiGraph
        networkx graph of full tracking data

    Returns
    -------
    Tuple[Dict[Any, int], Dict[int, List[int]]]
        A tuple containing:
        - A dictionary mapping node IDs to tracklet IDs.
        - A dictionary mapping each node ID to a list of its parent tracklet IDs.
    """
    track_id = 1
    visited_nodes = set()
    node_to_tid = {}
    parent_graph = defaultdict(list)

    for node in graph.nodes():
        if node in visited_nodes:
            continue

        start_node = node
        while graph.in_degree(start_node) == 1:
            predecessor = list(graph.predecessors(start_node))[0]
            if predecessor in visited_nodes:
                break
            start_node = predecessor

        current_tracklet = []
        temp_node = start_node
        while True:
            current_tracklet.append(temp_node)
            visited_nodes.add(temp_node)

            if graph.out_degree(temp_node) != 1:

                for child in graph.successors(temp_node):
                    parent_graph[child].append(temp_node)
                break

            successor = list(graph.successors(temp_node))[0]

            if graph.in_degree(successor) != 1:
                parent_graph[successor].append(temp_node)
                break

            temp_node = successor

        for node_id in current_tracklet:
            node_to_tid[node_id] = track_id

        track_id += 1

    track_graph = {
        node_to_tid[node_id]: [node_to_tid[node_id_] for node_id_ in parents]
        for node_id, parents in parent_graph.items()
    }

    return node_to_tid, track_graph
