# andromeda_drop.py
#
# Utility module for dropping rows or columns from a pandas DataFrame
# based on patterns found in cell values.
#
# Features:
#   - Drop ROWS that contain one or more patterns anywhere.
#   - Drop COLUMNS that contain one or more patterns anywhere.
#   - Drop BOTH rows and columns with a single call.
#   - Case-insensitive search: data and patterns are normalized to lowercase.
#   - Two modes:
#       - strict substring search  (regex=False, default)
#       - regex search            (regex=True)
#
#   API style matches trimming utilities:
#
#       drop_rows(df)["bmw"]
#       drop_rows(df)[["bmw", "audi"]]
#       drop_rows(df)[("bmw", True)]                 # regex=True
#       drop_rows(df)[("bmw", "audi", True)]         # many patterns + regex
#       drop_cols(df)[r"^unnamed", True]
#       drop_both(df)["bmw", "audi", True]


import pandas as pd


# =====================================================================
#  Core helpers
# =====================================================================

def _normalize_patterns(patterns):
    """
    Ensure patterns is a list of lowercase, stripped strings.
    """
    if patterns is None:
        return []
    if not isinstance(patterns, (list, tuple, set)):
        patterns = [patterns]
    return [str(p).lower().strip() for p in patterns]


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all values to lowercase strings for uniform, case-insensitive search.
    """
    return df.astype(str).apply(lambda col: col.str.lower())


def _drop_rows(df: pd.DataFrame, patterns, regex: bool = False) -> pd.DataFrame:
    """
    Drop rows where any cell matches any of the patterns.
    """
    df_copy = df.copy()
    patterns = _normalize_patterns(patterns)

    if not patterns or df_copy.empty:
        return df_copy.reset_index(drop=True)

    data = _prepare_data(df_copy)

    # Start with "no rows to drop"
    mask = pd.Series(False, index=df_copy.index)

    for p in patterns:
        if regex:
            m = data.apply(lambda col: col.str.contains(p, regex=True, na=False)).any(axis=1)
        else:
            # strict substring search (no regex syntax)
            m = data.apply(lambda col: col.str.contains(p, regex=False, na=False)).any(axis=1)
        mask |= m

    # Keep only rows that are NOT matched
    return df_copy.loc[~mask].reset_index(drop=True)


def _drop_cols(df: pd.DataFrame, patterns, regex: bool = False) -> pd.DataFrame:
    """
    Drop columns where any cell matches any of the patterns.
    """
    df_copy = df.copy()
    patterns = _normalize_patterns(patterns)

    if not patterns or df_copy.empty:
        return df_copy

    data = _prepare_data(df_copy)

    # Start with "no columns to drop"
    mask = pd.Series(False, index=df_copy.columns)

    for p in patterns:
        if regex:
            m = data.apply(lambda col: col.str.contains(p, regex=True, na=False))
        else:
            m = data.apply(lambda col: col.str.contains(p, regex=False, na=False))
        # Any match in a column → mark column for dropping
        mask |= m.any(axis=0)

    # Keep only columns that are NOT matched
    return df_copy.loc[:, ~mask]


def drop_by_pattern(
        df: pd.DataFrame,
        patterns,
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

    patterns : str or list/tuple/set of str
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
    result = df.copy()

    if not rows and not cols:
        return result

    if rows:
        result = _drop_rows(result, patterns, regex=regex)

    if cols and not result.empty:
        result = _drop_cols(result, patterns, regex=regex)

    return result


# =====================================================================
#  Proxy objects enabling clean syntax:
#      drop_rows(df)["pattern"]
#      drop_rows(df)[("pattern1", "pattern2", True)]
# =====================================================================

class _DropRowsProxy:
    """
    Proxy enabling:

        drop_rows(df)["bmw"]
        drop_rows(df)[["bmw", "audi"]]
        drop_rows(df)[("bmw", True)]                # regex=True
        drop_rows(df)[("bmw", "audi", True)]        # many patterns + regex flag
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, key):
        """
        If key is a tuple and the last element is bool → treated as regex flag.

        Examples:
            "bmw"                           → patterns=["bmw"], regex=False
            ["bmw", "audi"]                 → patterns=["bmw","audi"], regex=False
            ("bmw", True)                   → patterns=("bmw",), regex=True
            ("bmw", "audi", True)           → patterns=("bmw","audi"), regex=True
            ("bmw", "audi")                 → patterns=("bmw","audi"), regex=False
        """
        regex = False
        patterns = key

        if isinstance(key, tuple):
            # if last element is bool, treat as regex flag, rest as patterns
            if len(key) >= 2 and isinstance(key[-1], bool):
                regex = key[-1]
                patterns = key[:-1]
            else:
                patterns = key

        if not isinstance(patterns, (list, tuple, set)):
            patterns = [patterns]

        return drop_by_pattern(self.df, patterns, rows=True, cols=False, regex=regex)


class _DropColsProxy:
    """
    Proxy enabling:

        drop_cols(df)["unnamed"]
        drop_cols(df)[r"^unnamed", True]
        drop_cols(df)[("kba", "hersteller", True)]
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, key):
        regex = False
        patterns = key

        if isinstance(key, tuple):
            if len(key) >= 2 and isinstance(key[-1], bool):
                regex = key[-1]
                patterns = key[:-1]
            else:
                patterns = key

        if not isinstance(patterns, (list, tuple, set)):
            patterns = [patterns]

        return drop_by_pattern(self.df, patterns, rows=False, cols=True, regex=regex)


class _DropBothProxy:
    """
    Proxy enabling:

        drop_both(df)["bmw"]
        drop_both(df)[["bmw", "audi"]]
        drop_both(df)[("bmw", "audi", True)]
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, key):
        regex = False
        patterns = key

        if isinstance(key, tuple):
            if len(key) >= 2 and isinstance(key[-1], bool):
                regex = key[-1]
                patterns = key[:-1]
            else:
                patterns = key

        if not isinstance(patterns, (list, tuple, set)):
            patterns = [patterns]

        return drop_by_pattern(self.df, patterns, rows=True, cols=True, regex=regex)


# =====================================================================
#  Public entry points (same style as trim_top / trim_bot)
# =====================================================================

def drop_rows(df: pd.DataFrame) -> _DropRowsProxy:
    """Entry point for dropping rows by pattern."""
    return _DropRowsProxy(df)


def drop_cols(df: pd.DataFrame) -> _DropColsProxy:
    """Entry point for dropping columns by pattern."""
    return _DropColsProxy(df)


def drop_both(df: pd.DataFrame) -> _DropBothProxy:
    """Entry point for dropping both rows and columns by pattern."""
    return _DropBothProxy(df)
