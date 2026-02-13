from __future__ import annotations

from typing import Mapping

import pandas as pd


def apply_config_filters(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    season_start = int(config["season_start"])
    season_end = int(config["season_end"])
    positions = list(config.get("positions", []))

    season_mask = (df["season"] >= season_start) & (df["season"] <= season_end)
    if positions:
        season_mask &= df["position"].isin(positions)

    return df[season_mask].copy()
