# trim.py
#
# Structural tool for trimming the edges of a pandas DataFrame
# based on one or more anchor patterns in cell values.
#
# Single, explicit API:
#
#   from trim import trim
#
#   df2 = trim({
#       "data_frame": df,
#       "top": ("Insgesamt", False),   # (anchor, keep)
#       "bottom": ("SONSTIGE", True),
#       "left": ("Marke", True),
#       "right": ("2024", True),
#       "max_rows_to_scan": 30,
#       "max_cols_to_scan": 10,
#       "match_mode": "equals",       # or "contains" (default)
#   })
#
# There are no shortcuts/proxies here; everything is configured explicitly
# via a single config dict to keep the API clean and predictable.

from typing import Any, Dict, Optional, Tuple, TypedDict, Literal

import numpy as np
import pandas as pd

# =====================================================================
#  Public config type
# =====================================================================

MatchMode = Literal["contains", "equals"]


class TrimConfig(TypedDict, total=False):
    """
    Configuration for trim(...):

        data_frame       : pandas.DataFrame              (required)
        top              : (anchor, keep_bool) or None   (optional)
        bottom           : (anchor, keep_bool) or None   (optional)
        left             : (anchor, keep_bool) or None   (optional)
        right            : (anchor, keep_bool) or None   (optional)
        max_rows_to_scan : int >= 1, default 20          (optional)
        max_cols_to_scan : int >= 1, default 20          (optional)
        match_mode       : "contains" or "equals",
                           default "contains"            (optional)
    """
    data_frame: pd.DataFrame
    top: Optional[Tuple[str, bool]]
    bottom: Optional[Tuple[str, bool]]
    left: Optional[Tuple[str, bool]]
    right: Optional[Tuple[str, bool]]
    max_rows_to_scan: int
    max_cols_to_scan: int
    match_mode: MatchMode


_ALLOWED_KEYS = {
    "data_frame",
    "top",
    "bottom",
    "left",
    "right",
    "max_rows_to_scan",
    "max_cols_to_scan",
    "match_mode",
}


# =====================================================================
#  Validation helpers
# =====================================================================

def _require_dataframe(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Ensure cfg['data_frame'] exists and is a pandas.DataFrame.
    """
    if "data_frame" not in cfg:
        raise KeyError("'data_frame' key is required in trim(...) config")

    df = cfg["data_frame"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"'data_frame' must be a pandas.DataFrame, got {type(df).__name__}"
        )
    return df


def _validate_int(name: str, value: Any, minimum: int = 1) -> int:
    """
    Ensure the given value is an int and >= minimum.
    """
    if not isinstance(value, int):
        raise TypeError(f"'{name}' must be int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"'{name}' must be >= {minimum}, got {value}")
    return value


def _validate_side(name: str, value: Any) -> Optional[Tuple[str, bool]]:
    """
    Validate side anchor config for top/bottom/left/right.

    Allowed:
        None
        ("Insgesamt", False)
        ("Marke", True)

    Returns:
        None or (anchor_str, keep_bool)
    """
    if value is None:
        return None

    if not isinstance(value, (tuple, list)):
        raise TypeError(
            f"'{name}' must be a tuple/list (anchor, keep) or None, "
            f"got {type(value).__name__}"
        )

    if len(value) != 2:
        raise ValueError(
            f"'{name}' must be a 2-element tuple/list: (anchor, keep_bool), "
            f"got length {len(value)}"
        )

    anchor, keep = value
    anchor_str = str(anchor)
    if not isinstance(keep, bool):
        raise TypeError(
            f"'{name}' second element 'keep' must be bool, got {type(keep).__name__}"
        )

    return anchor_str, keep


def _validate_match_mode(value: Any) -> MatchMode:
    """
    Ensure match_mode is one of the supported values.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"'match_mode' must be a string 'contains' or 'equals', "
            f"got {type(value).__name__}"
        )

    mode = value.strip().lower()
    if mode not in ("contains", "equals"):
        raise ValueError(
            f"'match_mode' must be 'contains' or 'equals', got {value!r}"
        )

    return mode  # type: ignore[return-value]


def _validate_trim_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate dict-style config for trim({...}).

    Ensures:
        - only known keys are present
        - required 'data_frame' is provided
        - types and ranges for options are correct
        - default values are applied for missing fields
    """
    unknown = set(cfg.keys()) - _ALLOWED_KEYS
    if unknown:
        raise KeyError(f"Unknown configuration key(s) for trim: {sorted(unknown)}")

    df = _require_dataframe(cfg)

    resolved: Dict[str, Any] = {
        "data_frame": df,
        "top": None,
        "bottom": None,
        "left": None,
        "right": None,
        "max_rows_to_scan": 20,
        "max_cols_to_scan": 20,
        "match_mode": "contains",  # backward-compatible default
    }

    # sides
    if "top" in cfg:
        resolved["top"] = _validate_side("top", cfg["top"])
    if "bottom" in cfg:
        resolved["bottom"] = _validate_side("bottom", cfg["bottom"])
    if "left" in cfg:
        resolved["left"] = _validate_side("left", cfg["left"])
    if "right" in cfg:
        resolved["right"] = _validate_side("right", cfg["right"])

    # scan windows
    if "max_rows_to_scan" in cfg:
        resolved["max_rows_to_scan"] = _validate_int(
            "max_rows_to_scan",
            cfg["max_rows_to_scan"],
            minimum=1,
        )
    if "max_cols_to_scan" in cfg:
        resolved["max_cols_to_scan"] = _validate_int(
            "max_cols_to_scan",
            cfg["max_cols_to_scan"],
            minimum=1,
        )

    # match mode
    if "match_mode" in cfg:
        resolved["match_mode"] = _validate_match_mode(cfg["match_mode"])

    return resolved


# =====================================================================
#  Core helpers
# =====================================================================

def _row_mask(
        block: pd.DataFrame,
        anchor: str,
        match_mode: MatchMode,
) -> pd.Series:
    """
    Return a boolean Series marking rows that contain the anchor
    in ANY column of the given block.

    match_mode:
        "contains" : case-insensitive substring match
        "equals"   : case-insensitive exact match after strip()
    """
    if block.empty:
        return pd.Series(False, index=block.index)

    a = str(anchor).lower().strip()

    if match_mode == "contains":
        mask_per_col = block.apply(
            lambda col: col.astype(str)
            .str.lower()
            .str.contains(a, na=False)
        )
    else:  # "equals"
        mask_per_col = block.apply(
            lambda col: col.astype(str)
            .str.lower()
            .str.strip()
            .eq(a)
        )

    return mask_per_col.any(axis=1)


def _col_mask(
        block: pd.DataFrame,
        anchor: str,
        match_mode: MatchMode,
) -> pd.Series:
    """
    Return a boolean Series marking columns that contain the anchor
    in ANY of their values.

    match_mode:
        "contains" : case-insensitive substring match
        "equals"   : case-insensitive exact match after strip()
    """
    if block.empty:
        return pd.Series(False, index=block.columns)

    a = str(anchor).lower().strip()

    if match_mode == "contains":
        mask_per_col = block.astype(str).apply(
            lambda col: col.str.lower().str.contains(a, na=False)
        )
    else:  # "equals"
        mask_per_col = block.astype(str).apply(
            lambda col: col.str.lower().str.strip().eq(a)
        )

    return mask_per_col.any(axis=0)


# =====================================================================
#  Core implementation
# =====================================================================

def _trim_core(
        df: pd.DataFrame,
        *,
        top: Optional[Tuple[str, bool]],
        bottom: Optional[Tuple[str, bool]],
        left: Optional[Tuple[str, bool]],
        right: Optional[Tuple[str, bool]],
        max_rows_to_scan: int,
        max_cols_to_scan: int,
        match_mode: MatchMode,
) -> pd.DataFrame:
    """
    Core function implementing the trimming transformation.

    The original DataFrame is not modified; a copy is created and returned.
    Row indices are reset (RangeIndex) when rows are removed.
    Column labels are preserved; only column positions are sliced.
    """
    result = df.copy()

    # ---------------- TOP ----------------
    if top is not None and not result.empty:
        anchor, keep = top
        max_rows = min(max_rows_to_scan, len(result))
        head = result.head(max_rows)

        mask = _row_mask(head, anchor, match_mode)
        if mask.any():
            first_pos_in_head = int(np.where(mask.to_numpy())[0][0])
            start_row = first_pos_in_head + (0 if keep else 1)
            result = result.iloc[start_row:].reset_index(drop=True)

    # ---------------- BOTTOM ----------------
    if bottom is not None and not result.empty:
        anchor, keep = bottom
        max_rows = min(max_rows_to_scan, len(result))
        tail = result.tail(max_rows)

        mask = _row_mask(tail, anchor, match_mode)
        if mask.any():
            last_pos_in_tail = int(np.where(mask.to_numpy())[0][-1])
            last_label = tail.index[last_pos_in_tail]

            pos_in_result = result.index.get_loc(last_label)
            end_row = pos_in_result + (1 if keep else 0)
            result = result.iloc[:end_row].reset_index(drop=True)

    # ---------------- LEFT ----------------
    if left is not None and not result.empty:
        anchor, keep = left
        max_cols = min(max_cols_to_scan, result.shape[1])
        left_block = result.iloc[:, :max_cols]

        if left_block.shape[1] > 0:
            mask = _col_mask(left_block, anchor, match_mode)
            if mask.any():
                first_col_in_block = int(np.where(mask.to_numpy())[0][0])
                start_col = first_col_in_block + (0 if keep else 1)
                result = result.iloc[:, start_col:]

    # ---------------- RIGHT ----------------
    if right is not None and not result.empty:
        anchor, keep = right
        max_cols = min(max_cols_to_scan, result.shape[1])
        right_block = result.iloc[:, -max_cols:]

        if right_block.shape[1] > 0:
            mask = _col_mask(right_block, anchor, match_mode)
            if mask.any():
                last_col_in_block = int(np.where(mask.to_numpy())[0][-1])
                offset = result.shape[1] - right_block.shape[1]
                last_col = offset + last_col_in_block

                end_col = last_col + (1 if keep else 0)
                result = result.iloc[:, :end_col]

    return result


# =====================================================================
#  Public API
# =====================================================================

def trim(config: TrimConfig) -> pd.DataFrame:
    """
    Main public entry point.

    Expects a dict-like config with keys:

        "data_frame"       : pandas.DataFrame              (required)
        "top"              : (anchor, keep_bool) or None   (optional)
        "bottom"           : (anchor, keep_bool) or None   (optional)
        "left"             : (anchor, keep_bool) or None   (optional)
        "right"            : (anchor, keep_bool) or None   (optional)
        "max_rows_to_scan" : int >= 1, default 20          (optional)
        "max_cols_to_scan" : int >= 1, default 20          (optional)
        "match_mode"       : "contains" or "equals",
                             default "contains"            (optional)

    All behavior is configured explicitly via this single config dict.
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"trim(...) expects a dict as config, got {type(config).__name__}"
        )

    resolved = _validate_trim_config_dict(config)

    df = resolved["data_frame"]
    return _trim_core(
        df,
        top=resolved["top"],
        bottom=resolved["bottom"],
        left=resolved["left"],
        right=resolved["right"],
        max_rows_to_scan=resolved["max_rows_to_scan"],
        max_cols_to_scan=resolved["max_cols_to_scan"],
        match_mode=resolved["match_mode"],
    )
