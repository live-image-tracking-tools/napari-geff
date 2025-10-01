from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

import geff
import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray
from typing_extensions import NotRequired

DTypeStr = Literal[
    "double",
    "int",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "float32",
    "float64",
    "str",
]
Axes = Literal["t", "z", "y", "x"]


class DesignHints(TypedDict):
    display_vertical: NotRequired[str]
    display_horizontal: NotRequired[str]
    display_depth: NotRequired[str]


class GraphAttrs(TypedDict):
    nodes: NDArray[Any]
    edges: NDArray[Any]
    t: NDArray[Any]
    z: NDArray[Any]
    y: NDArray[Any]
    x: NDArray[Any]
    extra_node_props: dict[str, NDArray[Any]]
    edge_props: dict[str, NDArray[Any]]
    directed: bool
    axis_names: tuple[Axes, ...]
    axis_units: tuple[str, ...]
    axis_types: tuple[str, ...]  # Added for type declaration
    axis_scales: tuple[float | None, ...]
    axis_offset: tuple[float | None, ...]
    design_hints: NotRequired[DesignHints]


class ExampleNodeProps(TypedDict):
    position: DTypeStr


class ExampleEdgeProps(TypedDict):
    score: DTypeStr
    color: DTypeStr


def create_dummy_graph_props(
    node_dtype: DTypeStr,
    node_prop_dtypes: ExampleNodeProps,
    edge_prop_dtypes: ExampleEdgeProps,
    directed: bool,
) -> GraphAttrs:
    """Creates a dictionary of graph properties for testing."""
    axis_names: tuple[Axes, ...] = ("t", "z", "y", "x")
    axis_units = ("second", "nanometer", "nanometer", "nanometer")
    axis_types = ("time", "space", "space", "space")  # Added axis types
    axis_scales = (None, 0.5, 2.0, 1.0)
    axis_offset = (None, 0.0, 1.0, 3.0)

    nodes = np.array([10, 2, 127, 4, 5], dtype=node_dtype)
    t = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=node_prop_dtypes["position"])
    z = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=node_prop_dtypes["position"])
    y = np.array(
        [100.0, 200.0, 300.0, 400.0, 500.0], dtype=node_prop_dtypes["position"]
    )
    x = np.array([1.0, 0.1, 0.1, 0.1, 0.1], dtype=node_prop_dtypes["position"])

    edges = np.array([[10, 2], [2, 127], [2, 4], [4, 5]], dtype=node_dtype)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=edge_prop_dtypes["score"])
    colors = np.array([1, 2, 3, 4], dtype=edge_prop_dtypes["color"])

    return {
        "nodes": nodes,
        "edges": edges,
        "t": t,
        "z": z,
        "y": y,
        "x": x,
        "extra_node_props": {},
        "edge_props": {"score": scores, "color": colors},
        "directed": directed,
        "axis_names": axis_names,
        "axis_units": axis_units,
        "axis_types": axis_types,  # Added to returned dict,
        "axis_scales": axis_scales,
        "axis_offset": axis_offset,
    }


# Using a fixture instead of a function so the tmp_path fixture is automatically passed
# Implemented as a closure where tmp_path is the bound variable
@pytest.fixture
def path_w_expected_graph_props(
    tmp_path,
) -> Callable[
    [DTypeStr, ExampleNodeProps, ExampleEdgeProps, bool],
    tuple[Path, GraphAttrs],
]:
    """
    Fixture to write a geff graph to disk and return its path and properties.
    """

    def func(
        node_dtype: DTypeStr,
        node_prop_dtypes: ExampleNodeProps,
        edge_prop_dtypes: ExampleEdgeProps,
        directed: bool,
    ) -> tuple[Path, GraphAttrs]:
        graph_props = create_dummy_graph_props(
            node_dtype=node_dtype,
            node_prop_dtypes=node_prop_dtypes,
            edge_prop_dtypes=edge_prop_dtypes,
            directed=directed,
        )

        # Build graph with networkx
        graph = nx.DiGraph() if directed else nx.Graph()
        for idx, node in enumerate(graph_props["nodes"]):
            props = {
                name: prop_array[idx]
                for name, prop_array in graph_props["extra_node_props"].items()
            }
            graph.add_node(
                node,
                t=graph_props["t"][idx],
                z=graph_props["z"][idx],
                y=graph_props["y"][idx],
                x=graph_props["x"][idx],
                **props,
            )

        for idx, edge in enumerate(graph_props["edges"]):
            props = {
                name: prop_array[idx]
                for name, prop_array in graph_props["edge_props"].items()
            }
            graph.add_edge(*edge.tolist(), **props)

        path = tmp_path / "rw_consistency.zarr/graph"

        geff.write(
            graph,
            path,
            axis_names=list(graph_props["axis_names"]),
            axis_units=list(graph_props["axis_units"]),
            axis_types=list(graph_props["axis_types"]),
            axis_scales=list(graph_props["axis_scales"]),
            axis_offset=list(graph_props["axis_offset"]),
        )

        return path, graph_props

    return func
