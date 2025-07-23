import typing
from typing import Union

if typing.TYPE_CHECKING:
    import networkx as nx


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
            if val1 != val2 or type_mismatch:
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

        map1, map2 = {canonical_edge(e): e for e in edges1}, {
            canonical_edge(e): e for e in edges2
        }
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
