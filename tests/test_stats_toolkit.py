"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""
import unittest

# from stats_toolkit import mean, median, mode, variance, std_dev, quartiles, iqr_outliers, StatisticsAnalyzer


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        self.skipTest("TODO")

    def test_median_even_and_odd_lengths(self):
        self.skipTest("TODO")

    def test_mode_with_ties(self):
        self.skipTest("TODO")

    def test_empty_input_raises(self):
        self.skipTest("TODO")


class TestSpread(unittest.TestCase):
    def test_sample_vs_population_variance(self):
        self.skipTest("TODO")

    def test_std_dev_single_value(self):
        self.skipTest("TODO")


class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        self.skipTest("TODO")

    def test_detects_obvious_outlier(self):
        self.skipTest("TODO")


class TestLoading(unittest.TestCase):
    def test_invalid_and_missing_values_are_rejected(self):
        self.skipTest("TODO")


if __name__ == "__main__":
    unittest.main()
