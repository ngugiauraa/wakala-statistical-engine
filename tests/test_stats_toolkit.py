"""Unit tests for the statistics toolkit."""

import unittest
import tempfile
from pathlib import Path

from stats_toolkit import (
    mean,
    median,
    mode,
    variance,
    std_dev,
    quartiles,
    iqr_outliers,
    StatisticsAnalyzer,
    load_column,
)


class TestCentralTendency(unittest.TestCase):

    def test_mean_of_known_values(self):
        self.assertEqual(mean([2, 4, 6, 8]), 5)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([1, 3, 5]), 3)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_mode_with_ties(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1, 2])

    def test_empty_input_raises(self):
        functions = [mean, median, mode, quartiles, iqr_outliers]

        for function in functions:
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function([])


class TestSpread(unittest.TestCase):

    def test_sample_vs_population_variance(self):
        values = [2, 4, 6, 8]

        self.assertAlmostEqual(
            variance(values, sample=True),
            20 / 3
        )

        self.assertAlmostEqual(
            variance(values, sample=False),
            5
        )

    def test_std_dev_single_value(self):
        self.assertEqual(std_dev([5], sample=False), 0)

        with self.assertRaises(ValueError):
            std_dev([5], sample=True)


class TestOutliers(unittest.TestCase):

    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers([10, 10, 10, 10, 10])

        self.assertEqual(result["outliers"], [])

    def test_detects_obvious_outlier(self):
        result = iqr_outliers([10, 11, 12, 13, 14, 100])

        self.assertIn(100, result["outliers"])
        self.assertGreater(result["upper_fence"], 14)
        self.assertLess(result["upper_fence"], 100)


class TestLoading(unittest.TestCase):

    def test_invalid_and_missing_values_are_rejected(self):
        csv_content = (
            "date,agent_id,daily_deposits_kes\n"
            "2026-09-01,AG-101,1000\n"
            "2026-09-02,AG-102,\n"
            "2026-09-03,AG-103,n/a\n"
            "2026-09-04,AG-104,2500\n"
            "2026-09-05,AG-105,abc\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_deposits.csv"
            csv_path.write_text(csv_content, encoding="utf-8")

            valid_values, rejected = load_column(
                csv_path,
                "daily_deposits_kes"
            )

        self.assertEqual(valid_values, [1000.0, 2500.0])

        self.assertEqual(
            rejected,
            [
                (3, ""),
                (4, "n/a"),
                (6, "abc")
            ]
        )


if __name__ == "__main__":
    unittest.main()
