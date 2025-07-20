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
