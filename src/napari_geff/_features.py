from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr


def _pd_nullable_from_str(dtype_str: str) -> Any:
    """Map GEFF dtype strings to pandas nullable extension dtypes.

    Falls back to object if unknown.
    """
    s = dtype_str.lower()
    # integers
    if s == "int8":
        return pd.Int8Dtype()
    if s == "int16":
        return pd.Int16Dtype()
    if s == "int32":
        return pd.Int32Dtype()
    if s == "int64":
        return pd.Int64Dtype()
    # unsigned integers
    if s == "uint8":
        return pd.UInt8Dtype()
    if s == "uint16":
        return pd.UInt16Dtype()
    if s == "uint32":
        return pd.UInt32Dtype()
    if s == "uint64":
        return pd.UInt64Dtype()
    # floats
    if s == "float32":
        return pd.Float32Dtype()
    if s == "float64":
        return pd.Float64Dtype()
    # boolean
    if s == "bool":
        return pd.BooleanDtype()
    # strings
    if s == "string":
        return pd.StringDtype()

    # Fallback: use pandas object dtype
    return object


def _reorder_by_node_ids(
    values: np.ndarray,
    node_ids_in_store: np.ndarray,
    desired_order: Sequence[Any],
) -> np.ndarray:
    idx_map = {nid: i for i, nid in enumerate(node_ids_in_store.tolist())}
    indices = [idx_map[nid] for nid in desired_order]
    return values[indices]


def build_typed_node_features(
    store_path: str | Path,
    geff_metadata: Any,
    node_id_order: Sequence[Any],
) -> pd.DataFrame:
    """Build a features DataFrame from a geff zarr store (for use with Napari's tracks layer)."""
    z = zarr.open(store_path, mode="r")

    node_ids_arr = np.asarray(z["nodes"]["ids"][...])

    # Prefer metadata order
    meta = geff_metadata.node_props_metadata
    prop_names = list(meta.keys())

    # If there are no node properties, return only node_id
    if len(prop_names) == 0:
        return pd.DataFrame({"node_id": list(node_id_order)})

    # Otherwise, read property arrays from the canonical props group
    props_group = z["nodes"]["props"]

    data_cols: dict[str, pd.Series] = {}

    # Always include node_id in the requested order
    data_cols["node_id"] = pd.Series(
        list(node_id_order), dtype=pd.Int64Dtype()
    )

    for prop in prop_names:
        # Skip variable-length properties for napari features
        m = meta[prop]
        varlen = bool(m.varlength)
        dtype_str = m.dtype
        if varlen:
            continue

        values_arr = np.asarray(props_group[prop]["values"][...])

        # Optional missing mask (True means missing)
        missing_mask = None
        if "missing" in props_group[prop]:  # type: ignore[operator]
            missing_mask = np.asarray(props_group[prop]["missing"][...])

        # Reorder to match requested node_id order
        values_ord = _reorder_by_node_ids(
            values_arr, node_ids_arr, node_id_order
        )

        missing_ord = (
            _reorder_by_node_ids(
                missing_mask, node_ids_arr, node_id_order
            ).astype(bool)
            if missing_mask is not None
            else None
        )

        # Determine pandas nullable dtype
        dtype_for_cast = dtype_str or str(values_arr.dtype)
        pd_dtype = _pd_nullable_from_str(dtype_for_cast)

        # Create Series with the extension dtype, then apply mask
        s = pd.Series(values_ord)
        try:
            s = s.astype(pd_dtype)
        except TypeError:
            s = s.astype("object")

        if missing_ord is not None:
            # Assign pandas NA to missing positions
            s.loc[missing_ord] = pd.NA

        data_cols[prop] = s

    # Assemble DataFrame with node_id first
    col_order = ["node_id"] + [
        c for c in prop_names if c in data_cols and c != "node_id"
    ]
    df = pd.DataFrame(data_cols)
    print("Here")
    return df.loc[:, col_order]
