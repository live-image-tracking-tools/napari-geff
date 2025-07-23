from collections import defaultdict
from typing import Any, Union

import networkx as nx
from geff import GeffMetadata


def diff_nx_graphs(
    g1: Union["nx.Graph", "nx.DiGraph"],
    g2: Union["nx.Graph", "nx.DiGraph"],
    check_types: bool = True,
) -> list[tuple[str, ...]]:
    """
    Compares two NetworkX graphs and returns a detailed list of differences.

    Args:
        g1: The first graph to compare.
        g2: The second graph to compare.
        check_types: If True (default), attribute comparisons will also check
            for type equality (e.g., int(1) vs float(1.0)). If False, only
            values are compared.

    Returns:
        A list of tuples describing the differences. An empty list means
        the graphs are identical. The output format includes:
        - ('node_attribute_diff', node, key, (val1, type1), (val2, type2))
        - ('edge_attribute_diff', edge, key, (val1, type1), (val2, type2))
    """
    diffs = []

    # Check for differences in graph type
    is_directed_mismatch = g1.is_directed() != g2.is_directed()
    is_multi_mismatch = g1.is_multigraph() != g2.is_multigraph()
    if is_directed_mismatch or is_multi_mismatch:
        type1 = f"directed={g1.is_directed()}, multigraph={g1.is_multigraph()}"
        type2 = f"directed={g2.is_directed()}, multigraph={g2.is_multigraph()}"
        diffs.append(("graph_type", type1, type2))
        return diffs

    # Compare nodes
    nodes1, nodes2 = set(g1.nodes), set(g2.nodes)
    for node in nodes1 - nodes2:
        diffs.append(("node_missing_from_g2", node))
    for node in nodes2 - nodes1:
        diffs.append(("node_missing_from_g1", node))

    # Granular node attribute comparison
    for node in nodes1 & nodes2:
        attrs1, attrs2 = g1.nodes[node], g2.nodes[node]
        all_keys = set(attrs1.keys()) | set(attrs2.keys())
        for key in all_keys:
            val1, val2 = attrs1.get(key), attrs2.get(key)

            # Check for difference in value, and optionally type
            type_mismatch = check_types and type(val1) is not type(val2)
            if val1 is not val2 or type_mismatch:
                type1_str = type(val1).__name__ if key in attrs1 else "N/A"
                type2_str = type(val2).__name__ if key in attrs2 else "N/A"
                v1_rep = val1 if key in attrs1 else "<missing>"
                v2_rep = val2 if key in attrs2 else "<missing>"
                diffs.append(
                    (
                        "node_attribute_diff",
                        node,
                        key,
                        (v1_rep, type1_str),
                        (v2_rep, type2_str),
                    )
                )

    # Compare edges
    if g1.is_multigraph():
        edges1, edges2 = set(g1.edges(keys=True)), set(g2.edges(keys=True))
    else:
        edges1, edges2 = set(g1.edges()), set(g2.edges())

    if not g1.is_directed():

        def canonical_edge(edge):
            u, v, *key = edge
            if u > v:
                u, v = v, u
            return (u, v, *key)

        map1, map2 = (
            {canonical_edge(e): e for e in edges1},
            {canonical_edge(e): e for e in edges2},
        )
        canon_edges1, canon_edges2 = set(map1.keys()), set(map2.keys())
    else:
        map1, map2 = {e: e for e in edges1}, {e: e for e in edges2}
        canon_edges1, canon_edges2 = edges1, edges2

    for edge in canon_edges1 - canon_edges2:
        diffs.append(("edge_missing_from_g2", map1[edge]))
    for edge in canon_edges2 - canon_edges1:
        diffs.append(("edge_missing_from_g1", map2[edge]))

    # Granular edge attribute comparison
    for edge in canon_edges1 & canon_edges2:
        orig_edge1, orig_edge2 = map1[edge], map2[edge]
        attrs1 = (
            g1.get_edge_data(*orig_edge1)
            if g1.is_multigraph()
            else g1.get_edge_data(orig_edge1[0], orig_edge1[1])
        )
        attrs2 = (
            g2.get_edge_data(*orig_edge2)
            if g2.is_multigraph()
            else g2.get_edge_data(orig_edge2[0], orig_edge2[1])
        )

        all_keys = set(attrs1.keys()) | set(attrs2.keys())
        for key in all_keys:
            val1, val2 = attrs1.get(key), attrs2.get(key)

            # Check for difference in value, and optionally type
            type_mismatch = check_types and type(val1) is not type(val2)
            if val1 != val2 or type_mismatch:
                type1_str = type(val1).__name__ if key in attrs1 else "N/A"
                type2_str = type(val2).__name__ if key in attrs2 else "N/A"
                v1_rep = val1 if key in attrs1 else "<missing>"
                v2_rep = val2 if key in attrs2 else "<missing>"
                diffs.append(
                    (
                        "edge_attribute_diff",
                        edge,
                        key,
                        (v1_rep, type1_str),
                        (v2_rep, type2_str),
                    )
                )
    return diffs


def get_tracklets_nx(
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


def get_display_axes(
    geff_metadata: GeffMetadata,
) -> tuple[list[str], str | None]:
    """Get display axes from geff metadata.

    Inspects geff_metadata.axes and geff_metadata.display_hints
    to determine the display axes in the order of time, depth, vertical,
    horizontal. At most 4 spatiotemporal axes are returned, even if
    more are present, as napari tracks layer only supports 4 axes on
    top of track ID.

    Parameters
    ----------
    geff_metadata : GeffMetadata
        Metadata object containing axis information.

    Returns
    -------
    list[str]
        List of display axes names in the order of time, depth, vertical, horizontal.
    """
    axes = geff_metadata.axes
    time_axis_name = None
    spatial_axes_names = []
    for axis in axes:
        if axis.type == "time":
            time_axis_name = axis.name
        elif axis.type == "space":
            spatial_axes_names.append(axis.name)

    # if display hints are provided, we make sure our spatial axis names
    # are ordered accordingly
    display_axis_dict = {}
    if geff_metadata.display_hints:
        display_hints = geff_metadata.display_hints
        if display_hints.display_depth:
            display_axis_dict["depth"] = display_hints.display_depth
        if display_hints.display_vertical:
            display_axis_dict["vertical"] = display_hints.display_vertical
        if display_hints.display_horizontal:
            display_axis_dict["horizontal"] = display_hints.display_horizontal
    display_axes = []
    for axis_type in ["depth", "vertical", "horizontal"]:
        if axis_type in display_axis_dict:
            display_axes.append(display_axis_dict[axis_type])
            spatial_axes_names.remove(display_axis_dict[axis_type])
    display_axes = spatial_axes_names + display_axes
    # we always take the time axis if we have it
    if time_axis_name:
        display_axes.insert(0, time_axis_name)
    if len(display_axes) > 4:
        # if there are more than 4 axes, we only take the innermost spatial axes
        # but we always include the time axis
        display_axes = (
            display_axes[-4:]
            if not time_axis_name
            else [display_axes[0]] + display_axes[-3:]
        )
    return display_axes, time_axis_name
