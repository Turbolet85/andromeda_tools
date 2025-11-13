# andromeda_trim.py
#
# Utility module for trimming pandas DataFrames using anchor words.
#
# Features:
#   - Trim from TOP, BOTTOM, LEFT, or RIGHT.
#   - Anchor search is case-insensitive and whitespace-tolerant
#     (both anchor and data are normalized to lowercase, stripped).
#   - You can choose whether to keep or remove the row/column containing the anchor.
#   - Convenient syntax:
#
#       trim_top(df)["Insgesamt"]          # keep anchor row
#       trim_top(df)["Insgesamt", False]   # drop anchor row
#       trim_bot(df)["SONSTIGE"]
#       trim_left(df)["Marke", False]
#
#   Under the hood it uses a single core function `trim_df`.


import numpy as np
import pandas as pd


# =====================================================================
#  Core trimming function used internally and by proxy objects
# =====================================================================

def trim_df(
        df: pd.DataFrame,
        top=None,  # ("anchor", keep_bool) or None
        bottom=None,  # ("anchor", keep_bool) or None
        left=None,  # ("anchor", keep_bool) or None
        right=None,  # ("anchor", keep_bool) or None
        max_rows_to_scan: int = 20,
        max_cols_to_scan: int = 20,
) -> pd.DataFrame:
    """
    Universal DataFrame trimming function.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to be trimmed.

    top, bottom, left, right : tuple or None
        Each side can be:
            None
        or:
            ("anchor_string", keep_bool)

        where:
            keep_bool = True  → keep the row/column that contains the anchor
            keep_bool = False → remove the row/column that contains the anchor

    max_rows_to_scan : int
        How many rows from top/bottom to scan for the anchor.

    max_cols_to_scan : int
        How many columns from left/right to scan for the anchor.

    Returns
    -------
    pd.DataFrame
        Trimmed DataFrame (a copy; the original is not modified).
    """

    # Work on a copy to avoid accidental modification of the original
    result = df.copy()

    # -----------------------------------------------------------------
    # Helper: build a boolean row mask for rows that contain the anchor
    # in ANY column of the given block.
    #
    # Normalization:
    #   - anchor is converted to str, then lower-case, then stripped
    #   - all data in the block is converted to string, lower-cased
    #     and then searched with .str.contains()
    # -----------------------------------------------------------------
    def row_mask(block: pd.DataFrame, anchor) -> pd.Series:
        anchor = str(anchor).lower().strip()
        return block.apply(
            lambda col: col.astype(str).str.lower().str.contains(anchor, na=False)
        ).any(axis=1)

    # -----------------------------------------------------------------
    # Helper: build a boolean column mask for columns that contain the
    # anchor in ANY of their values.
    #
    # Same normalization logic as in row_mask.
    # -----------------------------------------------------------------
    def col_mask(block: pd.DataFrame, anchor) -> pd.Series:
        anchor = str(anchor).lower().strip()
        return block.astype(str).apply(
            lambda col: col.str.lower().str.contains(anchor, na=False)
        ).any(axis=0)

    # =================================================================
    #                           TRIM TOP
    # =================================================================
    if top is not None and not result.empty:
        anchor, keep = top

        # Scan only the first `max_rows_to_scan` rows for performance
        head = result.head(max_rows_to_scan)

        mask = row_mask(head, anchor)

        if mask.any():
            # Find the FIRST matching row position within `head`
            first_pos_in_head = np.where(mask)[0][0]
            first_label = head.index[first_pos_in_head]

            # Convert index label to positional index in the full DataFrame.
            # Using get_indexer to be safe even if index is non-unique.
            pos = result.index.get_indexer([first_label])[0]

            # keep=True  → start slicing from the anchor row
            # keep=False → start slicing from the row AFTER the anchor
            start_row = pos + (0 if keep else 1)

            result = result.iloc[start_row:].reset_index(drop=True)

    # =================================================================
    #                          TRIM BOTTOM
    # =================================================================
    if bottom is not None and not result.empty:
        anchor, keep = bottom

        # Scan only the last `max_rows_to_scan` rows
        tail = result.tail(max_rows_to_scan)

        mask = row_mask(tail, anchor)

        if mask.any():
            # Find the LAST matching row position within `tail`
            last_pos_in_tail = np.where(mask)[0][-1]
            last_label = tail.index[last_pos_in_tail]

            # Convert index label to positional index in the full DataFrame
            pos = result.index.get_indexer([last_label])[0]

            # keep=True  → keep anchor row (slice up to pos inclusive)
            # keep=False → slice up to the row BEFORE anchor
            end_pos = pos + (1 if keep else 0)

            # iloc stop is exclusive, so end_pos works correctly:
            #   keep=True  → [:pos+1] includes anchor row
            #   keep=False → [:pos]   excludes anchor row
            result = result.iloc[:end_pos].reset_index(drop=True)

    # =================================================================
    #                           TRIM LEFT
    # =================================================================
    if left is not None and not result.empty:
        anchor, keep = left

        # Scan only the first `max_cols_to_scan` columns
        left_block = result.iloc[:, :max_cols_to_scan]

        if left_block.shape[1] > 0:
            mask = col_mask(left_block, anchor)

            if mask.any():
                # First matching column position within `left_block`
                first_col_in_block = np.where(mask)[0][0]

                # keep=True  → keep anchor column (start from it)
                # keep=False → drop anchor column (start from next)
                start_col = first_col_in_block + (0 if keep else 1)

                result = result.iloc[:, start_col:]

    # =================================================================
    #                           TRIM RIGHT
    # =================================================================
    if right is not None and not result.empty:
        anchor, keep = right

        # Scan only the last `max_cols_to_scan` columns
        right_block = result.iloc[:, -max_cols_to_scan:]

        if right_block.shape[1] > 0:
            mask = col_mask(right_block, anchor)

            if mask.any():
                # Offset to convert from relative column index in `right_block`
                # to absolute column index in `result`
                offset = result.shape[1] - right_block.shape[1]

                # Last matching column position within `right_block`
                last_col_in_block = np.where(mask)[0][-1]

                # Absolute position in the full DataFrame
                last_col = offset + last_col_in_block

                # keep=True  → slice including anchor column
                # keep=False → slice up to the column BEFORE anchor
                end_col = last_col + (1 if keep else 0)

                # iloc stop is exclusive:
                #   keep=True  → [:last_col+1] includes anchor column
                #   keep=False → [:last_col]   excludes anchor column
                result = result.iloc[:, :end_col]

    return result


# =====================================================================
#  Proxy objects enabling clean syntax:
#      trim_top(df)["Anchor"]
#      trim_top(df)["Anchor", False]
# =====================================================================

class _TopProxy:
    """
    Proxy class that enables syntax like:

        trim_top(df)["Anchor"]          # keep anchor row
        trim_top(df)["Anchor", False]   # drop anchor row

    It stores:
        - the DataFrame
        - max_rows_to_scan for top trimming
    """

    def __init__(self, df: pd.DataFrame, max_rows_to_scan: int = 20):
        self.df = df
        self.max_rows_to_scan = max_rows_to_scan

    def __getitem__(self, key):
        """
        key can be:
            "Anchor"            → interpreted as (anchor="Anchor", keep=True)
            ("Anchor", False)   → interpreted as (anchor="Anchor", keep=False)
        """
        if isinstance(key, tuple):
            anchor, keep = key
        else:
            anchor, keep = key, True

        return trim_df(
            self.df,
            top=(anchor, keep),
            max_rows_to_scan=self.max_rows_to_scan,
        )


class _BottomProxy:
    """
    Proxy for bottom trimming with syntax:

        trim_bot(df)["Anchor"]
        trim_bot(df)["Anchor", False]
    """

    def __init__(self, df: pd.DataFrame, max_rows_to_scan: int = 20):
        self.df = df
        self.max_rows_to_scan = max_rows_to_scan

    def __getitem__(self, key):
        if isinstance(key, tuple):
            anchor, keep = key
        else:
            anchor, keep = key, True

        return trim_df(
            self.df,
            bottom=(anchor, keep),
            max_rows_to_scan=self.max_rows_to_scan,
        )


class _LeftProxy:
    """
    Proxy for left-side column trimming:

        trim_left(df)["Anchor"]
        trim_left(df)["Anchor", False]
    """

    def __init__(self, df: pd.DataFrame, max_cols_to_scan: int = 20):
        self.df = df
        self.max_cols_to_scan = max_cols_to_scan

    def __getitem__(self, key):
        if isinstance(key, tuple):
            anchor, keep = key
        else:
            anchor, keep = key, True

        return trim_df(
            self.df,
            left=(anchor, keep),
            max_cols_to_scan=self.max_cols_to_scan,
        )


class _RightProxy:
    """
    Proxy for right-side column trimming:

        trim_right(df)["Anchor"]
        trim_right(df)["Anchor", False]
    """

    def __init__(self, df: pd.DataFrame, max_cols_to_scan: int = 20):
        self.df = df
        self.max_cols_to_scan = max_cols_to_scan

    def __getitem__(self, key):
        if isinstance(key, tuple):
            anchor, keep = key
        else:
            anchor, keep = key, True

        return trim_df(
            self.df,
            right=(anchor, keep),
            max_cols_to_scan=self.max_cols_to_scan,
        )


# =====================================================================
#  Convenience wrapper functions
# =====================================================================

def trim_top(df: pd.DataFrame, max_rows_to_scan: int = 20) -> _TopProxy:
    """
    Entry point for trimming from the top.

    Usage:
        trim_top(df)["Anchor"]          # keep row with anchor
        trim_top(df)["Anchor", False]   # drop row with anchor
    """
    return _TopProxy(df, max_rows_to_scan=max_rows_to_scan)


def trim_bot(df: pd.DataFrame, max_rows_to_scan: int = 20) -> _BottomProxy:
    """
    Entry point for trimming from the bottom.

    Usage:
        trim_bot(df)["Anchor"]
        trim_bot(df)["Anchor", False]
    """
    return _BottomProxy(df, max_rows_to_scan=max_rows_to_scan)


def trim_left(df: pd.DataFrame, max_cols_to_scan: int = 20) -> _LeftProxy:
    """
    Entry point for trimming from the left (columns).

    Usage:
        trim_left(df)["Anchor"]
        trim_left(df)["Anchor", False]
    """
    return _LeftProxy(df, max_cols_to_scan=max_cols_to_scan)


def trim_right(df: pd.DataFrame, max_cols_to_scan: int = 20) -> _RightProxy:
    """
    Entry point for trimming from the right (columns).

    Usage:
        trim_right(df)["Anchor"]
        trim_right(df)["Anchor", False]
    """
    return _RightProxy(df, max_cols_to_scan=max_cols_to_scan)
