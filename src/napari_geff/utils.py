from typing import Optional

import napari
import pandas as pd


def get_tracks_layer_df(
    tracks_layer: napari.layers.Tracks,
) -> (pd.DataFrame, list[str], list[str]):
    """Get a dataframe of track data from a napari Tracks layer, with labelled columns.
    Note that this also adds a node_id to the final column of the DataFrame,

    Parameters
    ----------
    tracks_layer :
        The napari Tracks layer.

    Returns
    -------
    tracks_layer_df
        DataFrame containing the track data with columns for napari_track_id, time, and spatial axes.
    axis_names : list[str]
        List of axis names in the tracks layer.
    axis_types : list[str]
        List of axis types in the tracks layer, e.g. ['time', 'space', ...].
    """

    if "geff_metadata" in tracks_layer.metadata:
        axis_names = [
            axis.name for axis in tracks_layer.metadata["geff_metadata"].axes
        ]
        axis_types = [
            axis.type for axis in tracks_layer.metadata["geff_metadata"].axes
        ]
        tracks_layer_df = tracks_layer.features
        assert (
            "node_id" in tracks_layer_df.columns
        ), "Tracks layer must have a node_id column"
        return tracks_layer_df, axis_names, axis_types
    else:
        axis_names = list(tracks_layer.axis_labels)
        axis_types = ["time"] + ["space"] * (len(axis_names) - 1)
        tracks_layer_df = pd.DataFrame(
            tracks_layer.data,
            columns=["napari_track_id"] + axis_names,
            # napari assumes the first two cols are tid and t
        )
        # Since no metadata, use index as node id. TODO maybe allow user to specify a node id in features or metadata
        tracks_layer_df["node_id"] = tracks_layer_df.index
        return tracks_layer_df, axis_names, axis_types


def edges_from_tracks_layer(
    tracks_layer: napari.layers.Tracks,
) -> pd.DataFrame:
    """Creates a minimal edge list dataframe from the tracks layer data.
    Requires napari_track_id, and node_id_columns.

    Parameters
    ----------
    tracks_layer_data :
        DataFrame containing the track data assuming columns are track_id, time, and spatial axes in that order.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the edges of the tracks.
    """

    tracks_layer_data, axis_names, axis_types = get_tracks_layer_df(
        tracks_layer
    )

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
    if tracks_layer.graph:

        tracklet_extrema = {
            tid: {
                "first": tracklet_df["node_id"].iloc[
                    0
                ],  # it works because they're sorteed by time
                "last": tracklet_df["node_id"].iloc[-1],
            }
            for tid, tracklet_df in tracks_layer_data.groupby(
                "napari_track_id"
            )
        }

        for (
            daughter_tracklet_id,
            parent_tracklet_ids,
        ) in tracks_layer.graph.items():
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


def add_tracks_layer_from_sample(
    viewer: napari.Viewer,
    tracks_layer_data: pd.DataFrame,
    tracklets_graph: Optional[dict] = None,
) -> None:
    """Add a tracks layer to the napari viewer from a sample Tracks layer
    using the minimal amount of data required.

    Parameters
    ----------
    viewer :
        The napari viewer to add the layer to.
    tracks_layer_data :
        DataFrame containing the track data assuming columns are track_id, time, and spatial axes. in that order
    tracklets_graph :
        The tracks graph that napari expects for merges and splits
    """
    viewer.add_tracks(
        data=tracks_layer_data,
        name="Tracks",
        graph=tracklets_graph,
    )
