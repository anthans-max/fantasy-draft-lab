import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
TARGET_PLAYERS_PER_SEASON = 26

np.random.seed(RANDOM_SEED)

DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / "sample_players.csv"
OUTPUT_PARQUET = DATA_DIR / "fantasy_players_2019_2025.parquet"

df_seed = pd.read_csv(INPUT_CSV)

seasons = sorted(df_seed["season"].unique())
positions = df_seed["position"].unique()

rows = []

for season in seasons:
    season_df = df_seed[df_seed["season"] == season]

    for pos in positions:
        pos_df = season_df[season_df["position"] == pos]

        if pos_df.empty:
            continue

        samples_needed = max(
            1,
            TARGET_PLAYERS_PER_SEASON // len(positions)
        )

        for _ in range(samples_needed):
            base = pos_df.sample(1).iloc[0]

            games = np.random.randint(12, 18)

            rushing = max(0, int(base["rushing_yards"] * np.random.uniform(0.7, 1.3)))
            receiving = max(0, int(base["receiving_yards"] * np.random.uniform(0.7, 1.3)))
            receptions = max(0, int(base["receptions"] * np.random.uniform(0.7, 1.3)))
            tds = max(0, int(base["touchdowns"] * np.random.uniform(0.6, 1.4)))

            fantasy_ppr = (
                rushing * 0.1 +
                receiving * 0.1 +
                receptions * 1.0 +
                tds * 6
            )

            fantasy_half = (
                rushing * 0.1 +
                receiving * 0.1 +
                receptions * 0.5 +
                tds * 6
            )

            rows.append({
                "player": f"{base['player']} ({season})",
                "position": pos,
                "season": season,
                "adp_overall": round(np.random.uniform(1, 200), 1),
                "games_played": games,
                "rushing_yards": rushing,
                "receiving_yards": receiving,
                "receptions": receptions,
                "touchdowns": tds,
                "fantasy_points_ppr": round(fantasy_ppr, 2),
                "fantasy_points_half_ppr": round(fantasy_half, 2),
                "fantasy_points_per_game_ppr": round(fantasy_ppr / games, 2),
                "fantasy_points_per_game_half_ppr": round(fantasy_half / games, 2),
            })

df_out = pd.DataFrame(rows)

print("Generated rows:", len(df_out))
print("Seasons:", df_out["season"].min(), "→", df_out["season"].max())

df_out.to_parquet(OUTPUT_PARQUET, index=False)
print("Wrote:", OUTPUT_PARQUET)
