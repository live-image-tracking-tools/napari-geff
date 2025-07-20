import napari
import pandas as pd


def get_tracks_layer_df(tracks_layer: napari.layers.Tracks) -> pd.DataFrame:
    """Get a dataframe of track data from a napari Tracks layer, with labelled columns.

    Parameters
    ----------
    tracks_layer :
        The napari Tracks layer.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the track data.
    """

    if "geff_metadata" in tracks_layer.metadata:
        axis_names = [
            axis.name for axis in tracks_layer.metadata["geff_metadata"].axes
        ]
        assert "time" in [
            axis.type for axis in tracks_layer.metadata["geff_metadata"].axes
        ], "Tracks layer must have a time axis"
        tracks_layer_df = tracks_layer.features[
            ["napari_track_id"] + axis_names
        ]
        return tracks_layer_df
    else:
        axis_names = list(tracks_layer.axis_labels)
        tracks_layer_df = pd.DataFrame(
            tracks_layer.data,
            columns=["napari_track_id"] + axis_names,
            # napari assumes the first two cols are tid and t
        )
        # Since no metadata, use index as node id. TODO maybe allow user to specify a node id in features or metadata
        tracks_layer_df["node_id"] = tracks_layer_df.index
