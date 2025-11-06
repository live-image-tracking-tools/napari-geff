"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import networkx as nx
import numpy as np
import pandas as pd
from geff import write

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def _to_python_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalar types to Python scalars, normalizing booleans."""
    if isinstance(value, np.bool_ | bool):
        return np.bool_(bool(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_edge_attributes(
    edge_props: dict[Any, dict[str, Any]],
) -> dict[Any, dict[str, Any]]:
    """Return edge attribute mapping with scalar values coerced via `_to_python_scalar`."""
    prepared: dict[Any, dict[str, Any]] = {}
    for edge, props in edge_props.items():
        if isinstance(edge, tuple):
            normalized_edge = tuple(_to_python_scalar(part) for part in edge)
        else:
            normalized_edge = edge
        prepared[normalized_edge] = {
            key: _to_python_scalar(val) for key, val in props.items()
        }
    return prepared


def _ensure_numpy_bool_attributes(graph: nx.Graph) -> None:
    """Get everthing into `numpy.bool_`"""
    for _, data in graph.nodes(data=True):
        for key, value in list(data.items()):
            if isinstance(value, bool):
                data[key] = np.bool_(value)
    for _, _, data in graph.edges(data=True):
        for key, value in list(data.items()):
            if isinstance(value, bool):
                data[key] = np.bool_(value)


def write_tracks(path: str, data: Any, meta: dict) -> list[str]:
    layer_metadata = meta["metadata"]
    features = meta["features"]
    tracklets_graph = meta["graph"]

    if "geff_metadata" in layer_metadata:
        axis_names = [
            axis.name for axis in layer_metadata["geff_metadata"].axes
        ]
        axis_types = [
            axis.type for axis in layer_metadata["geff_metadata"].axes
        ]
        tracks_layer_df = features
        assert (
            "node_id" in tracks_layer_df.columns
        ), "Tracks layer must have a node_id column"
    else:
        axis_names = list(meta["axis_labels"])
        axis_types = ["time"] + ["space"] * (len(axis_names) - 1)
        tracks_layer_df = pd.DataFrame(
            data,
            columns=["napari_track_id"] + axis_names,
            # napari assumes the first two cols are tid and t
        )

        tracks_layer_df = pd.concat([tracks_layer_df, features], axis=1)

        # Since no metadata, use index as node id. TODO maybe allow user to specify a node id in features or metadata
        tracks_layer_df["node_id"] = tracks_layer_df.index

    edge_df = get_edge_df(
        tracks_layer_data=tracks_layer_df,
        tracklets_graph=tracklets_graph,
        axis_names=axis_names,
        axis_types=axis_types,
    )

    nx_graph = create_nx_graph(
        tracks_layer_data=tracks_layer_df,
        edges_df=edge_df,
        axis_names=axis_names,
        axis_types=axis_types,
        edge_properties=layer_metadata.get("edge_properties", None),
        base_graph=layer_metadata.get("nx_graph"),
    )
    _ensure_numpy_bool_attributes(nx_graph)
    write(
        nx_graph,
        path,
        layer_metadata.get("geff_metadata", None),
        axis_names=axis_names,
        axis_types=axis_types,
    )

    return [path]


def get_edge_df(
    tracks_layer_data: pd.DataFrame,
    tracklets_graph: dict[int, list[int]],
    axis_names: list[str],
    axis_types: list[str],
) -> pd.DataFrame:
    edges = []

    # Get the name of the time axis
    t_axis = axis_names[axis_types.index("time")]

    tracks_layer_data.sort_values(by=["napari_track_id", t_axis], inplace=True)

    # First do the intra-tracklet edges
    for _tid, track_df in tracks_layer_data.groupby("napari_track_id"):
        nodes = track_df["node_id"].tolist()

        # add source and target nodes
        for node1, node2 in zip(nodes[:-1], nodes[1:], strict=False):
            edges.append({"source": node1, "target": node2})

    # Next get the splits and merges from the graph dictionary
    if tracklets_graph:
        tracklet_extrema = {
            tid: {
                "first": tracklet_df["node_id"].iloc[
                    0
                ],  # it works because they're sorted by time
                "last": tracklet_df["node_id"].iloc[-1],
            }
            for tid, tracklet_df in tracks_layer_data.groupby(
                "napari_track_id"
            )
        }

        for (
            daughter_tracklet_id,
            parent_tracklet_ids,
        ) in tracklets_graph.items():
            for parent_tracklet_id in parent_tracklet_ids:
                edges.append(
                    {
                        "source": tracklet_extrema[parent_tracklet_id]["last"],
                        "target": tracklet_extrema[daughter_tracklet_id][
                            "first"
                        ],
                    }
                )

    return pd.DataFrame(edges)


def create_nx_graph(
    tracks_layer_data: pd.DataFrame,
    edges_df: pd.DataFrame,
    axis_names: list[str],
    axis_types: list[str],
    edge_properties: dict[str, Any] | None = None,
    base_graph: nx.DiGraph | None = None,
) -> nx.DiGraph:
    """
    Create a networkx directed graph from a napari Tracks layer.
    """

    # Ignore napari-only bookkeeping columns when rebuilding node attrs.
    skip_columns: set[str] = {"napari_track_id"}
    if axis_names:
        skip_columns.update(axis_names)

    def _row_to_attrs(row: pd.Series) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for column, value in row.items():
            if column in skip_columns or column == "node_id":
                continue
            if pd.isna(value):
                continue
            attrs[column] = _to_python_scalar(value)
        return attrs

    tracks_layer_data = tracks_layer_data.copy()

    # Start from the existing geff graph if we have one otherwise, build fresh.
    if base_graph is not None:
        nx_graph = base_graph.copy(as_view=False)
    else:
        nx_graph = nx.DiGraph()

    # Track desired edges using python scalars so they match the target graph.
    desired_edges: set[tuple[Any, Any]] = set()
    if not edges_df.empty:
        for _, row in edges_df.iterrows():
            desired_edges.add(
                (
                    _to_python_scalar(row["source"]),
                    _to_python_scalar(row["target"]),
                )
            )

    # Allow for graph modifications when writing out
    for node_id in tracks_layer_data["node_id"]:
        node_id_py = _to_python_scalar(node_id)
        if not nx_graph.has_node(node_id_py):
            nx_graph.add_node(node_id_py)

    existing_edges = set(nx_graph.edges())
    if base_graph is None:
        # Fresh graph: align edge set exactly with the desired tracks edges.
        for edge in desired_edges - existing_edges:
            nx_graph.add_edge(*edge)
        for edge in existing_edges - desired_edges:
            nx_graph.remove_edge(*edge)
    else:
        for edge in desired_edges:
            # Allow for graph modifications when writing out
            if edge not in existing_edges:
                nx_graph.add_edge(*edge)

    node_attr_updates = {}
    for _, row in tracks_layer_data.iterrows():
        node_id = _to_python_scalar(row["node_id"])
        attrs = _row_to_attrs(row)
        if attrs:
            node_attr_updates.setdefault(node_id, {}).update(attrs)

        # Preserve axis-aligned coords
        positional_attrs: dict[str, Any] = {}
        if axis_names:
            for name in axis_names:
                value = row.get(name)
                if pd.isna(value):
                    continue
                positional_attrs[name] = _to_python_scalar(value)
        if positional_attrs:
            node_attr_updates.setdefault(node_id, {}).update(positional_attrs)

    if node_attr_updates:
        nx.set_node_attributes(nx_graph, node_attr_updates)

    if edge_properties:
        prepared_edge_props = _prepare_edge_attributes(edge_properties)
        nx.set_edge_attributes(nx_graph, prepared_edge_props)

    return nx_graph
