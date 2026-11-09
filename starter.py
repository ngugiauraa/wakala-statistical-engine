import csv
import math
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""

    valid_values = []
    rejected = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            raw_value = row.get(column, "").strip()

            try:
                value = float(raw_value)

                if not math.isfinite(value):
                    raise ValueError

                valid_values.append(value)

            except ValueError:
                rejected.append((row_number, raw_value))

    return valid_values, rejected


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate the mean of an empty list.")

    total = sum(values)
    count = len(values)

    return total / count


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate the median of an empty list.")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n % 2 == 1:
        return sorted_values[n // 2]
    else:
        mid1 = sorted_values[n // 2 - 1]
        mid2 = sorted_values[n // 2]
        return (mid1 + mid2) / 2


def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""

    if not values:
        raise ValueError("Cannot calculate the mode of an empty list.")

    frequencies = {}

    for value in values:
        frequencies[value] = frequencies.get(value, 0) + 1

    highest_frequency = max(frequencies.values())

    if highest_frequency == 1:
        return []

    modes = [
        value
        for value, frequency in frequencies.items()
        if frequency == highest_frequency
    ]

    return sorted(modes)


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""

    if not values:
        raise ValueError("Cannot calculate variance of an empty list.")

    if sample and len(values) < 2:
        raise ValueError("Sample variance requires at least two values.")

    average = mean(values)

    squared_deviations = [
        (value - average) ** 2
        for value in values
    ]

    denominator = len(values) - 1 if sample else len(values)

    return sum(squared_deviations) / denominator


def std_dev(values: list[float], sample: bool = True) -> float:
    return variance(values, sample) ** 0.5


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3) using linear interpolation (Hyndman-Fan Type 7)."""

    if not values:
        raise ValueError("Cannot calculate quartiles of an empty list.")

    sorted_values = sorted(values)
    n = len(sorted_values)

    def percentile(p: float) -> float:
        position = (n - 1) * p

        lower_index = math.floor(position)
        upper_index = math.ceil(position)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        fraction = position - lower_index

        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]

        return lower_value + fraction * (upper_value - lower_value)

    q1 = percentile(0.25)
    q2 = percentile(0.50)
    q3 = percentile(0.75)

    return q1, q2, q3


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, lower and upper fences, and outlying values."""

    if not values:
        raise ValueError("Cannot detect outliers in an empty list.")

    if k < 0:
        raise ValueError("The outlier multiplier k cannot be negative.")

    q1, q2, q3 = quartiles(values)

    iqr = q3 - q1

    lower_fence = q1 - (k * iqr)
    upper_fence = q3 + (k * iqr)

    outliers = [
        value
        for value in values
        if value < lower_fence or value > upper_fence
    ]

    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outliers": outliers
    }


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes statistical analysis."""

    def __init__(self, values: list[float], name: str = "values") -> None:
        if not values:
            raise ValueError("Cannot analyze an empty dataset.")

        self.values = list(values)
        self.name = name

        # Sort the dataset only once.
        self.sorted_values = sorted(self.values)

        # Preserve rejected CSV entries.
        self.rejected = []

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        values, rejected = load_column(path, column)

        analyzer = cls(values, column)
        analyzer.rejected = rejected

        return analyzer

    def _median_sorted(self) -> float:
        """Calculate median using the cached sorted values."""

        n = len(self.sorted_values)
        mid = n // 2

        if n % 2 == 1:
            return self.sorted_values[mid]

        return (
            self.sorted_values[mid - 1] +
            self.sorted_values[mid]
        ) / 2

    def _quartiles_sorted(self) -> tuple[float, float, float]:
        """Calculate Type 7 quartiles using cached sorted values."""

        values = self.sorted_values
        n = len(values)

        def percentile(p: float) -> float:
            position = (n - 1) * p

            lower_index = math.floor(position)
            upper_index = math.ceil(position)

            if lower_index == upper_index:
                return values[lower_index]

            fraction = position - lower_index

            return (
                values[lower_index] +
                fraction * (
                    values[upper_index] - values[lower_index]
                )
            )

        return (
            percentile(0.25),
            percentile(0.50),
            percentile(0.75)
        )

    def summary(self) -> dict:
        """Return all descriptive statistics for this dataset."""

        count = len(self.values)

        average = mean(self.values)
        middle = self._median_sorted()
        modes = mode(self.values)

        sample_variance = variance(self.values, sample=True)
        sample_std_dev = sample_variance ** 0.5

        q1, q2, q3 = self._quartiles_sorted()

        iqr = q3 - q1

        lower_fence = q1 - (1.5 * iqr)
        upper_fence = q3 + (1.5 * iqr)

        outliers = [
            value for value in self.values
            if value < lower_fence or value > upper_fence
        ]

        return {
            "name": self.name,
            "count": count,
            "mean": average,
            "median": middle,
            "mode": modes,
            "variance": sample_variance,
            "std_dev": sample_std_dev,
            "min": self.sorted_values[0],
            "max": self.sorted_values[-1],
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outliers": outliers
        }

    def without_outliers(self) -> "StatisticsAnalyzer":
        """Return a new analyzer with IQR outliers removed."""

        stats = self.summary()
        outlier_values = stats["outliers"]

        cleaned_values = [
            value for value in self.values
            if value not in outlier_values
        ]

        if not cleaned_values:
            raise ValueError("Removing outliers would leave an empty dataset.")

        analyzer = StatisticsAnalyzer(cleaned_values, self.name)

        analyzer.rejected = self.rejected.copy()

        return analyzer

    def report(self) -> str:
        """Generate a statistical report for this dataset."""

        stats = self.summary()

        lines = [
            f"=== Statistics report: {self.name} ===",
            f"Valid values      : {stats['count']}",
            f"Rejected values   : {len(self.rejected)}",
            ""
        ]

        for row_number, raw_value in self.rejected:
            lines.append(
                f"  row {row_number}: '{raw_value}' "
                "(invalid numeric value)"
            )

        lines.extend([
            "",
            f"Mean              : {stats['mean']:.2f}",
            f"Median            : {stats['median']:.2f}",
            f"Mode              : {stats['mode'] if stats['mode'] else 'none'}",
            f"Variance (sample) : {stats['variance']:.2f}",
            f"Std dev (sample)  : {stats['std_dev']:.2f}",
            f"Minimum           : {stats['min']:.2f}",
            f"Maximum           : {stats['max']:.2f}",
            f"Q1                : {stats['q1']:.2f}",
            f"Q3                : {stats['q3']:.2f}",
            f"IQR               : {stats['iqr']:.2f}",
            "",
            f"Lower fence       : {stats['lower_fence']:.2f}",
            f"Upper fence       : {stats['upper_fence']:.2f}",
            f"Outliers          : {stats['outliers']}"
        ])

        return "\n".join(lines)

def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Compare our statistical calculations against NumPy."""

    import numpy as np

    values = analyzer.values

    results = {}

    # Mean
    numpy_mean = np.mean(values)
    results["mean"] = math.isclose(
        mean(values),
        numpy_mean,
        rel_tol=1e-9
    )

    # Median
    numpy_median = np.median(values)
    results["median"] = math.isclose(
        median(values),
        numpy_median,
        rel_tol=1e-9
    )

    # Sample variance: ddof=1 means denominator is n - 1.
    numpy_variance = np.var(values, ddof=1)
    results["variance"] = math.isclose(
        variance(values, sample=True),
        numpy_variance,
        rel_tol=1e-9
    )

    # Sample standard deviation
    numpy_std_dev = np.std(values, ddof=1)
    results["std_dev"] = math.isclose(
        std_dev(values, sample=True),
        numpy_std_dev,
        rel_tol=1e-9
    )

    # Quartiles: NumPy's linear interpolation method.
    numpy_q1, numpy_q3 = np.percentile(
        values,
        [25, 75],
        method="linear"
    )

    q1, _, q3 = quartiles(values)

    results["q1"] = math.isclose(
        q1, numpy_q1, rel_tol=1e-9
    )

    results["q3"] = math.isclose(
        q3, numpy_q3, rel_tol=1e-9
    )

    return results


if __name__ == "__main__":
    analyzer = StatisticsAnalyzer([1, 2, 3, 4, 100])

    print("Original:", analyzer.summary())

    cleaned = analyzer.without_outliers()

    print("Without outliers:", cleaned.summary())
    print("Original still intact:", analyzer.values)