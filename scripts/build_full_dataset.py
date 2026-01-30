from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl  # pip install nflreadpy


YEARS = list(range(2019, 2026))  # 2019–2025 inclusive
MIN_GAMES_PLAYED = 4

OUT_PATH = Path("data") / "fantasy_players_2019_2025.parquet"


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    # Pull regular season weekly stats via nflreadpy and aggregate to season totals.
    try:
        raw = nfl.load_player_stats(seasons=YEARS, summary_level="week")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch weekly data from nflreadpy: {e}") from e

    df = raw.to_pandas()

    # --- Map columns defensively (nflverse schemas can vary) ---
    name_col = _pick_col(df, ["player_name", "player_display_name", "player", "name"])
    pos_col = _pick_col(df, ["position", "pos"])
    season_col = _pick_col(df, ["season", "year"])
    season_type_col = _pick_col(df, ["season_type"])
    game_id_col = _pick_col(df, ["game_id", "gameid", "gsis_id"])
    week_col = _pick_col(df, ["week", "week_id"])

    rush_yds_col = _pick_col(df, ["rushing_yards", "rush_yards", "rsh_yds", "rush_yds"])
    rec_yds_col = _pick_col(df, ["receiving_yards", "rec_yards", "rec_yds"])
    rec_col = _pick_col(df, ["receptions", "rec", "recs"])

    rush_td_col = _pick_col(df, ["rushing_tds", "rush_tds", "rush_td", "rsh_tds"])
    rec_td_col = _pick_col(df, ["receiving_tds", "rec_tds", "rec_td"])

    required = {
        "player": name_col,
        "position": pos_col,
        "season": season_col,
        "game_id_or_week": game_id_col or week_col,
        "rushing_yards": rush_yds_col,
        "receiving_yards": rec_yds_col,
        "receptions": rec_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(
            "Could not find required columns in nfl_data_py weekly data. "
            f"Missing mappings for: {missing}. "
            f"Available columns (sample): {sorted(list(df.columns))[:80]}"
        )

    if game_id_col is None and week_col is None:
        raise RuntimeError(
            "Could not find a game id or week column to compute games_played. "
            f"Available columns (sample): {sorted(list(df.columns))[:80]}"
        )

    if season_type_col:
        df = df[df[season_type_col] == "REG"].copy()

    game_key = df[game_id_col].astype(str) if game_id_col else df[week_col].astype(str)

    base = pd.DataFrame(
        {
            "player": df[name_col].astype(str),
            "position": df[pos_col].astype(str),
            "season": df[season_col].astype(int),
            "game_key": game_key,
            "rushing_yards": pd.to_numeric(df[rush_yds_col], errors="coerce").fillna(0).astype(int),
            "receiving_yards": pd.to_numeric(df[rec_yds_col], errors="coerce").fillna(0).astype(int),
            "receptions": pd.to_numeric(df[rec_col], errors="coerce").fillna(0).astype(int),
        }
    )

    # Touchdowns = rushing TDs + receiving TDs (matches your existing schema semantics)
    rush_tds = pd.to_numeric(df[rush_td_col], errors="coerce").fillna(0) if rush_td_col else 0
    rec_tds = pd.to_numeric(df[rec_td_col], errors="coerce").fillna(0) if rec_td_col else 0
    base["touchdowns"] = (rush_tds + rec_tds).astype(int)

    # Keep only positions your app expects
    base = base[base["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    # Restrict to target seasons
    base = base[base["season"].isin(YEARS)].copy()

    # Aggregate weekly -> season totals per player
    grouped = (
        base.groupby(["player", "position", "season"], as_index=False)
        .agg(
            games_played=("game_key", "nunique"),
            rushing_yards=("rushing_yards", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receptions=("receptions", "sum"),
            touchdowns=("touchdowns", "sum"),
        )
        .copy()
    )

    # Filter noise: games_played >= 4
    out = grouped[grouped["games_played"] >= MIN_GAMES_PLAYED].copy()

    # ADP not included in seasonal stats → leave as NaN for now
    out["adp_overall"] = np.nan

    # Fantasy points (PPR + half PPR)
    out["fantasy_points_ppr"] = (
        out["rushing_yards"] * 0.1
        + out["receiving_yards"] * 0.1
        + out["receptions"] * 1.0
        + out["touchdowns"] * 6.0
    )

    out["fantasy_points_half_ppr"] = (
        out["rushing_yards"] * 0.1
        + out["receiving_yards"] * 0.1
        + out["receptions"] * 0.5
        + out["touchdowns"] * 6.0
    )

    gp = out["games_played"].replace(0, np.nan)
    out["fantasy_points_per_game_ppr"] = (out["fantasy_points_ppr"] / gp).round(3)
    out["fantasy_points_per_game_half_ppr"] = (out["fantasy_points_half_ppr"] / gp).round(3)

    # Order columns exactly like your existing sample schema
    out = out[
        [
            "player",
            "position",
            "season",
            "adp_overall",
            "games_played",
            "rushing_yards",
            "receiving_yards",
            "receptions",
            "touchdowns",
            "fantasy_points_ppr",
            "fantasy_points_half_ppr",
            "fantasy_points_per_game_ppr",
            "fantasy_points_per_game_half_ppr",
        ]
    ].copy()

    # Stable sort for reproducible output
    out.sort_values(["season", "position", "fantasy_points_ppr"], ascending=[True, True, False], inplace=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    # --- Print validation summary ---
    print("Wrote:", OUT_PATH)
    print("Rows:", len(out))
    print("Season range:", out["season"].min(), "→", out["season"].max())
    print("\nRows by season:")
    print(out["season"].value_counts().sort_index().to_string())
    print("\nRows by position:")
    print(out["position"].value_counts().to_string())
    print(f"\n% missing adp_overall: {out['adp_overall'].isna().mean() * 100:.1f}%")

    print("\nTop 10 by PPR (2025):")
    print(
        out[out["season"] == 2025]
        .head(10)[["player", "position", "fantasy_points_ppr", "games_played"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
