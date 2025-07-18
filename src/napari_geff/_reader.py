"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
import geff
import numpy as np
from geff.utils import validate
from geff import GeffMetadata
import zarr
import networkx as nx
from collections import defaultdict
import pandas as pd

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    try:
        validate(path)
    except Exception:
        return None

    graph = zarr.open(path, mode="r")

    # graph attrs validation
    # Raises pydantic.ValidationError or ValueError
    meta = GeffMetadata(**graph.attrs)
    if meta.position_attr is None and meta.axis_names is None:
        return None
    if not meta.directed:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    path = paths[0]

    nx_graph = geff.read_nx(path, validate = False)
    node_to_tid, track_graph = get_tracklets(nx_graph)

    #points = np.array(
    #    [[node_to_tid[node_id], data['t'], data['y'], data['x']] for node_id, data in G_hela.nodes(data=True)])
    #points = pd.DataFrame(points, columns=['track_id', 't', 'y', 'x'])
    #points['track_id'] = points['track_id'].astype(int)

    if "axis_names" in nx_graph.graph:
        axis_names = list(nx_graph.graph["axis_names"])
        tracks = pd.DataFrame(
            [[node_to_tid[node_id], data['t']] + [data[axis_name] for axis_name in axis_names] for node_id, data in nx_graph.nodes(data=True)]
        )

    else:
        position_attr = nx_graph.graph["position_attr"]
        tracks = pd.DataFrame(
            [[node_to_tid[node_id], data['t']] + data[position_attr] for node_id, data in
             nx_graph.nodes(data=True)]
        )
        position_ndim = tracks.ndim - 2 # because one for t and one for track_id
        axis_names = [f"axis_{i}" for i in range(position_ndim)]

    tracks.columns = ['track_id', 't'] + axis_names
    tracks.sort_values(by=['track_id', 't'], inplace=True)
    tracks['track_id'] = tracks['track_id'].astype(int)

    points = tracks[["t"] + axis_names].values
    metadata = {"nx_graph": nx_graph}

    return [
        (tracks, {"graph": track_graph, "name": "Tracks", "metadata": metadata}, "tracks"),
        (points, {"name": "Points", "metadata": {"nx_graph": metadata}}, "points")
    ]


def get_tracklets(graph: nx.DiGraph):
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
        node_to_tid[node_id]: [node_to_tid[node_id_]
                               for node_id_ in parents]
        for node_id, parents in parent_graph.items()
    }

    return node_to_tid, track_graph