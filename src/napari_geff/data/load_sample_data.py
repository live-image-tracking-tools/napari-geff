from pathlib import Path

from napari_geff._reader import reader_function


def load_sample_data():
    script_path = Path(__file__).resolve()
    path = (
        script_path.parent.parent.parent
        / "examples/graph_labels_seg.zarr/tracks"
    )
    path = str(path)
    layers = reader_function(path)
    return layers


def load_sample_data_3d():
    script_path = Path(__file__).resolve()
    path = script_path.parent.parent.parent / "examples/graph_3d.zarr/tracks"
    path = str(path)
    layers = reader_function(path)
    return layers
