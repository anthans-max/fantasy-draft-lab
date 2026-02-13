from __future__ import annotations

from typing import Iterable, Mapping, Tuple


DEFAULT_SCORING_FORMAT = "Half-PPR"
ALLOWED_SCORING_FORMATS = ("Half-PPR", "Full PPR")
DEFAULT_POSITIONS = ("QB", "RB", "WR", "TE")


def normalize_scoring_format(value: str | None) -> str:
    if value in ALLOWED_SCORING_FORMATS:
        return value

    text = (value or "").strip().lower().replace("_", " ").replace("-", " ")
    if text in {"full ppr", "ppr"}:
        return "Full PPR"
    if text in {"half ppr", "half"}:
        return "Half-PPR"
    return DEFAULT_SCORING_FORMAT


def normalize_season_range(
    value: int | Iterable[int] | None, min_season: int, max_season: int
) -> Tuple[int, int]:
    if min_season > max_season:
        min_season, max_season = max_season, min_season

    if value is None:
        return min_season, max_season

    if isinstance(value, int):
        season = min(max(value, min_season), max_season)
        return season, season

    values = list(value)
    if len(values) < 2:
        if len(values) == 1:
            season = int(values[0])
            season = min(max(season, min_season), max_season)
            return season, season
        return min_season, max_season

    start = int(values[0])
    end = int(values[1])
    if start > end:
        start, end = end, start
    start = min(max(start, min_season), max_season)
    end = min(max(end, min_season), max_season)
    return start, end


def normalize_positions(
    values: Iterable[str] | None,
    fallback_to_all: bool = True,
) -> list[str]:
    ordered_positions = list(DEFAULT_POSITIONS)
    if values is None:
        return ordered_positions if fallback_to_all else []

    cleaned = {
        str(value).strip().upper() for value in values if str(value).strip()
    }
    selected = [position for position in ordered_positions if position in cleaned]
    if not selected and fallback_to_all:
        return ordered_positions
    return selected


def normalize_config(
    raw_config: Mapping[str, object] | None,
    min_season: int,
    max_season: int,
    fallback_positions_to_all: bool = True,
) -> dict[str, object]:
    source = dict(raw_config or {})

    season_start, season_end = normalize_season_range(
        source.get("season_range"), min_season=min_season, max_season=max_season
    )
    if "season_start" in source or "season_end" in source:
        start_value = source.get("season_start", season_start)
        end_value = source.get("season_end", season_end)
        season_start, season_end = normalize_season_range(
            (start_value, end_value),
            min_season=min_season,
            max_season=max_season,
        )

    min_seasons_played = int(source.get("min_seasons_played", 2))
    min_seasons_played = max(min_seasons_played, 1)

    return {
        "scoring_format": normalize_scoring_format(
            str(source.get("scoring_format", DEFAULT_SCORING_FORMAT))
        ),
        "season_start": season_start,
        "season_end": season_end,
        "season_range": (season_start, season_end),
        "positions": normalize_positions(
            source.get("positions"), fallback_to_all=fallback_positions_to_all
        ),
        "min_seasons_played": min_seasons_played,
    }


def scoring_columns(scoring_format: str) -> Tuple[str, str]:
    use_full_ppr = normalize_scoring_format(scoring_format) == "Full PPR"
    points_column = "fantasy_points_ppr" if use_full_ppr else "fantasy_points_half_ppr"
    ppg_column = (
        "fantasy_points_per_game_ppr"
        if use_full_ppr
        else "fantasy_points_per_game_half_ppr"
    )
    return points_column, ppg_column
