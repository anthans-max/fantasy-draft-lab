import inspect
import unittest

import pandas as pd

from src.filters import apply_config_filters


class ApplyConfigFiltersTests(unittest.TestCase):
    def test_filter_uses_explicit_config_object(self) -> None:
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C", "D"],
                "position": ["QB", "RB", "WR", "QB"],
                "season": [2020, 2021, 2022, 2023],
            }
        )
        config_one = {
            "season_start": 2020,
            "season_end": 2022,
            "positions": ["QB", "WR"],
        }
        config_two = {
            "season_start": 2023,
            "season_end": 2023,
            "positions": ["QB"],
        }

        result_one = apply_config_filters(df, config_one)
        result_two = apply_config_filters(df, config_two)

        self.assertEqual(result_one["player"].tolist(), ["A", "C"])
        self.assertEqual(result_two["player"].tolist(), ["D"])

    def test_function_signature_requires_config_argument(self) -> None:
        params = list(inspect.signature(apply_config_filters).parameters)
        self.assertEqual(params, ["df", "config"])


if __name__ == "__main__":
    unittest.main()
