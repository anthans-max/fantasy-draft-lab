import unittest

from src.config import DEFAULT_POSITIONS, normalize_config


class NormalizeConfigTests(unittest.TestCase):
    def test_normalizes_scoring_seasons_positions_and_min_seasons(self) -> None:
        config = normalize_config(
            {
                "scoring_format": "ppr",
                "season_range": (2025, 2019),
                "positions": ["wr", "qb", "WR", "invalid"],
                "min_seasons_played": 0,
            },
            min_season=2019,
            max_season=2025,
            fallback_positions_to_all=False,
        )

        self.assertEqual(config["scoring_format"], "Full PPR")
        self.assertEqual(config["season_start"], 2019)
        self.assertEqual(config["season_end"], 2025)
        self.assertEqual(config["season_range"], (2019, 2025))
        self.assertEqual(config["positions"], ["QB", "WR"])
        self.assertEqual(config["min_seasons_played"], 1)

    def test_position_fallback_to_all_is_optional(self) -> None:
        no_fallback = normalize_config(
            {"positions": []},
            min_season=2020,
            max_season=2024,
            fallback_positions_to_all=False,
        )
        with_fallback = normalize_config(
            {"positions": []},
            min_season=2020,
            max_season=2024,
            fallback_positions_to_all=True,
        )

        self.assertEqual(no_fallback["positions"], [])
        self.assertEqual(with_fallback["positions"], list(DEFAULT_POSITIONS))


if __name__ == "__main__":
    unittest.main()
