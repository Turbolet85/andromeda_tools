# drop_utils.py
#
# Structural tool for dropping rows/columns from a pandas DataFrame
# based on presence of one or more patterns in cell values.
#
# Single, explicit API:
#
#   from drop_utils import drop
#
#   df2 = drop({
#       "data_frame": df,                 # required
#       "patterns": ["bmw", "audi"],      # required (str or sequence of str)
#       "regex": False,                   # optional, default False (substring search)
#       "rows": True,                     # optional, default True
#       "cols": False,                    # optional, default False
#   })
#
# There are no shortcuts/proxies here; everything is configured explicitly
# via a single config dict to keep the API clean and predictable.

from typing import Any, Dict, Iterable, List, Sequence, TypedDict, Union

import pandas as pd


# =====================================================================
#  Public config type
# =====================================================================

class DropConfig(TypedDict, total=False):
    """
    Configuration for drop(...):

        data_frame : pandas.DataFrame           (required)
        patterns   : str or sequence of str     (required)
        regex      : bool, default False
        rows       : bool, default True
        cols       : bool, default False
    """
    data_frame: pd.DataFrame
    patterns: Union[str, Sequence[str]]
    regex: bool
    rows: bool
    cols: bool


_ALLOWED_KEYS = {
    "data_frame",
    "patterns",
    "regex",
    "rows",
    "cols",
}


# =====================================================================
#  Validation helpers
# =====================================================================

def _require_dataframe(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Ensure cfg['data_frame'] exists and is a pandas.DataFrame.
    """
    if "data_frame" not in cfg:
        raise KeyError("'data_frame' key is required in drop(...) config")

    df = cfg["data_frame"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"'data_frame' must be a pandas.DataFrame, got {type(df).__name__}"
        )
    return df


def _normalize_patterns_raw(patterns: Any) -> List[str]:
    """
    Normalize patterns input (which may be a single value or an iterable)
    into a non-empty list of raw strings (no lowercasing yet).
    """
    if isinstance(patterns, (list, tuple, set)):
        vals = list(patterns)
    else:
        vals = [patterns]

    if not vals:
        raise ValueError("'patterns' must contain at least one element")

    result: List[str] = [str(p) for p in vals]
    return result


def _validate_patterns(cfg: Dict[str, Any]) -> List[str]:
    """
    Ensure 'patterns' is provided and normalize it into a list of strings.
    """
    if "patterns" not in cfg:
        raise KeyError("'patterns' key is required in drop(...) config")

    pats = _normalize_patterns_raw(cfg["patterns"])
    return pats


def _validate_bool(name: str, value: Any) -> bool:
    """
    Ensure the given value is a bool.
    """
    if not isinstance(value, bool):
        raise TypeError(f"'{name}' must be bool, got {type(value).__name__}")
    return value


def _validate_drop_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate dict-style config for drop({...}).

    Keys:
        data_frame : pandas.DataFrame           (required)
        patterns   : str or iterable of str     (required)
        regex      : bool, default False
        rows       : bool, default True
        cols       : bool, default False
    """
    unknown = set(cfg.keys()) - _ALLOWED_KEYS
    if unknown:
        raise KeyError(f"Unknown configuration key(s) for drop: {sorted(unknown)}")

    df = _require_dataframe(cfg)
    patterns = _validate_patterns(cfg)

    # defaults
    resolved: Dict[str, Any] = {
        "data_frame": df,
        "patterns": patterns,
        "regex": False,
        "rows": True,
        "cols": False,
    }

    # regex
    if "regex" in cfg:
        resolved["regex"] = _validate_bool("regex", cfg["regex"])

    # rows / cols
    if "rows" in cfg:
        resolved["rows"] = _validate_bool("rows", cfg["rows"])
    if "cols" in cfg:
        resolved["cols"] = _validate_bool("cols", cfg["cols"])

    return resolved


# =====================================================================
#  Core helpers
# =====================================================================

def _normalize_patterns(patterns: Iterable[str]) -> List[str]:
    """
    Normalize patterns to a list of lowercase, stripped strings.
    """
    return [str(p).lower().strip() for p in patterns]


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all values to lowercase strings for uniform, case-insensitive search.
    """
    return df.astype(str).apply(lambda col: col.str.lower())


def _drop_rows(
        df: pd.DataFrame,
        patterns: Iterable[str],
        regex: bool = False,
) -> pd.DataFrame:
    """
    Drop rows where any cell matches any of the patterns.
    """
    df_copy = df.copy()
    pats = _normalize_patterns(patterns)

    if not pats or df_copy.empty:
        return df_copy.reset_index(drop=True)

    data = _prepare_data(df_copy)

    # Start with "no rows to drop"
    mask = pd.Series(False, index=df_copy.index)

    for p in pats:
        if regex:
            matches = data.apply(
                lambda col: col.str.contains(p, regex=True, na=False)
            ).any(axis=1)
        else:
            # strict substring search (no regex syntax)
            matches = data.apply(
                lambda col: col.str.contains(p, regex=False, na=False)
            ).any(axis=1)
        mask |= matches

    # Keep only rows that are NOT matched
    return df_copy.loc[~mask].reset_index(drop=True)


def _drop_cols(
        df: pd.DataFrame,
        patterns: Iterable[str],
        regex: bool = False,
) -> pd.DataFrame:
    """
    Drop columns where any cell matches any of the patterns.
    """
    df_copy = df.copy()
    pats = _normalize_patterns(patterns)

    if not pats or df_copy.empty:
        return df_copy

    data = _prepare_data(df_copy)

    # Start with "no columns to drop"
    mask = pd.Series(False, index=df_copy.columns)

    for p in pats:
        if regex:
            matches = data.apply(
                lambda col: col.str.contains(p, regex=True, na=False)
            )
        else:
            matches = data.apply(
                lambda col: col.str.contains(p, regex=False, na=False)
            )

        # Any match in a column → mark column for dropping
        mask |= matches.any(axis=0)

    # Keep only columns that are NOT matched
    return df_copy.loc[:, ~mask]


def _drop_by_pattern_core(
        df: pd.DataFrame,
        patterns: Iterable[str],
        rows: bool = True,
        cols: bool = False,
        regex: bool = False,
) -> pd.DataFrame:
    """
    Core function: drop rows/columns that contain given pattern(s).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    patterns : iterable of str
        Patterns to search for (case-insensitive).
        If regex=False → treated as plain substrings.
        If regex=True  → treated as regular expressions.

    rows : bool
        If True, drop rows containing any of the patterns.

    cols : bool
        If True, drop columns containing any of the patterns.

    regex : bool
        If True, interpret patterns as regular expressions.
        If False, use strict substring search (no regex syntax).

    Returns
    -------
    pd.DataFrame
        New DataFrame with matching rows/columns removed.
    """
    rows = _validate_bool("rows", rows)
    cols = _validate_bool("cols", cols)
    regex = _validate_bool("regex", regex)

    result = df.copy()

    if not rows and not cols:
        # nothing to drop, just return a copy
        return result.reset_index(drop=True)

    if rows:
        result = _drop_rows(result, patterns, regex=regex)

    if cols and not result.empty:
        result = _drop_cols(result, patterns, regex=regex)

    return result


# =====================================================================
#  Public API
# =====================================================================

def drop(config: DropConfig) -> pd.DataFrame:
    """
    Main public entry point.

    Expects a dict-like config with keys:

        "data_frame" : pandas.DataFrame          (required)
        "patterns"   : str or iterable of str    (required)

        "regex"      : bool, default False
        "rows"       : bool, default True
        "cols"       : bool, default False

    Behavior:
        - rows=True  → drop rows containing any pattern.
        - cols=True  → drop columns containing any pattern.
        - Both True  → apply rows then columns.
        - All searches are case-insensitive.
        - regex=False → substring search (no regex syntax).
        - regex=True  → regular expressions.
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"drop(...) expects a dict as config, got {type(config).__name__}"
        )

    resolved = _validate_drop_config_dict(config)

    df = resolved["data_frame"]
    return _drop_by_pattern_core(
        df,
        patterns=resolved["patterns"],
        rows=resolved["rows"],
        cols=resolved["cols"],
        regex=resolved["regex"],
    )
