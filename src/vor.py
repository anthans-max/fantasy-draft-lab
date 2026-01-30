from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import pandas as pd


DEFAULT_REPLACEMENT_RANKS: Dict[str, int] = {
    "QB": 12,
    "RB": 24,
    "WR": 24,
    "TE": 12,
}


def _infer_ppg_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in (
        "fantasy_points_per_game_ppr",
        "fantasy_points_per_game_half_ppr",
        "fantasy_points_per_game",
        "ppg",
    ):
        if candidate in df.columns:
            return candidate
    return None


def _infer_points_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in (
        "fantasy_points_ppr",
        "fantasy_points_half_ppr",
        "fantasy_points",
        "points",
    ):
        if candidate in df.columns:
            return candidate
    return None


def ensure_ppg_column(
    df: pd.DataFrame,
    ppg_column: str,
    points_column: Optional[str] = None,
    games_column: str = "games_played",
) -> pd.DataFrame:
    df = df.copy()
    if ppg_column in df.columns and df[ppg_column].notna().any():
        return df

    if points_column is None:
        points_column = _infer_points_column(df)

    if points_column and points_column in df.columns and games_column in df.columns:
        games = df[games_column].where(df[games_column] > 0, pd.NA)
        df[ppg_column] = (df[points_column] / games).fillna(0)
    else:
        df[ppg_column] = 0

    return df


def add_draft_round(
    df: pd.DataFrame,
    league_teams: int = 12,
    season_col: str = "season",
    position_col: str = "position",
    ppg_column: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()

    if "draft_pick" in df.columns:
        df["draft_pick_used"] = df["draft_pick"]
        df["adp_used"] = pd.NA
        df["adp_source"] = "Draft Pick"
        base_values = df["draft_pick"]
    else:
        adp_column = None
        if "adp" in df.columns:
            adp_column = "adp"
        elif "adp_overall" in df.columns:
            adp_column = "adp_overall"

        if adp_column and df[adp_column].notna().any():
            df["draft_pick_used"] = pd.NA
            df["adp_used"] = df[adp_column]
            df["adp_source"] = "ADP"
            base_values = df[adp_column]
        else:
            if ppg_column is None:
                ppg_column = _infer_ppg_column(df) or "ppg"
            df = ensure_ppg_column(df, ppg_column)
            rank = df.groupby([season_col, position_col])[ppg_column].rank(
                method="first", ascending=False
            )
            df["draft_pick_used"] = pd.NA
            df["adp_used"] = rank * 2
            df["adp_source"] = "Demo ADP"
            base_values = df["adp_used"]

    df["draft_round"] = base_values.apply(
        lambda value: math.ceil(value / league_teams)
        if pd.notna(value)
        else pd.NA
    )

    def bucket_round(value: float) -> str:
        if pd.isna(value):
            return "Unknown"
        value = max(int(value), 1)
        return "20+" if value > 20 else str(value)

    df["draft_round_bucket"] = df["draft_round"].apply(bucket_round)
    return df


def add_vor(
    df: pd.DataFrame,
    ppg_column: str,
    replacement_ranks: Optional[Dict[str, int]] = None,
    season_col: str = "season",
    position_col: str = "position",
) -> pd.DataFrame:
    df = ensure_ppg_column(df, ppg_column)
    df = df.copy()

    replacement_ranks = replacement_ranks or DEFAULT_REPLACEMENT_RANKS
    default_rank = max(replacement_ranks.values()) if replacement_ranks else 12

    df["ppg_for_vor"] = df[ppg_column].fillna(0)
    df["replacement_rank"] = (
        df[position_col].map(replacement_ranks).fillna(default_rank).astype(int)
    )

    group_cols = [season_col, position_col]
    df["rank_in_group"] = df.groupby(group_cols)["ppg_for_vor"].rank(
        method="first", ascending=False
    )
    df["group_size"] = df.groupby(group_cols)["ppg_for_vor"].transform("size")
    df["effective_replacement_rank"] = df[["replacement_rank", "group_size"]].min(axis=1)

    replacement_rows = df.loc[
        df["rank_in_group"] == df["effective_replacement_rank"],
        group_cols + ["ppg_for_vor"],
    ].rename(columns={"ppg_for_vor": "replacement_ppg"})
    replacement_rows = replacement_rows.drop_duplicates(subset=group_cols)

    df = df.merge(replacement_rows, on=group_cols, how="left")
    df["replacement_ppg"] = df["replacement_ppg"].fillna(0)
    df["vor"] = (df["ppg_for_vor"] - df["replacement_ppg"]).clip(lower=0)

    return df


def self_check() -> Dict[str, bool]:
    data = pd.DataFrame(
        {
            "player": ["A", "B", "C", "D"],
            "position": ["QB", "QB", "RB", "RB"],
            "season": [2024, 2024, 2024, 2024],
            "ppg": [20.0, 10.0, 15.0, 5.0],
        }
    )
    ranks = {"QB": 1, "RB": 2}
    result = add_vor(data, ppg_column="ppg", replacement_ranks=ranks)
    qb_replacement = result.loc[result["player"] == "A", "replacement_ppg"].iloc[0]
    rb_replacement = result.loc[result["player"] == "C", "replacement_ppg"].iloc[0]
    return {
        "qb_replacement_rank": math.isclose(qb_replacement, 20.0),
        "rb_replacement_rank": math.isclose(rb_replacement, 5.0),
        "vor_clamped": result["vor"].min() >= 0,
    }
