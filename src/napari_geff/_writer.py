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
import pandas as pd
from geff import write

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


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
    )
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
) -> nx.DiGraph:
    """
    Create a networkx directed graph from a napari Tracks layer.
    """

    nx_graph = nx.from_pandas_edgelist(
        edges_df, create_using=nx.DiGraph(), edge_attr=None
    )  # Don't pass in edge attrs here, otherwise you build the wrong graph
    node_axis_properties = (
        tracks_layer_data.loc[
            :,
            [
                column
                for column in tracks_layer_data
                if "napari_track_id" not in column
            ],
        ]
        .set_index("node_id")
        .apply(lambda row: row.dropna().to_dict(), axis=1)
        .to_dict()
    )
    nx.set_node_attributes(nx_graph, node_axis_properties)

    if edge_properties:
        nx.set_edge_attributes(nx_graph, edge_properties)

    return nx_graph
