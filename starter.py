"""
Every calculation implemented with plain python.
NumPy is imported ONLY in compare_with_numpy(), to verify the results.
"""
import csv
import math
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
    valid_values = []
    rejected = []
    
    if not path.exists():
        raise FileNotFoundError(f"Target data file not found at path: {path}")
        
    with open(path, mode='r', encoding='utf-8') as f:
        # Use DictReader to dynamically locate the requested column
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):  # Row 1 is header
            raw_value = row.get(column)
            
            if raw_value is None:
                rejected.append((row_number, "MISSING_COLUMN"))
                continue
                
            stripped = raw_value.strip()
            
            # Catch standard empty expressions or textual placeholders
            if stripped == "" or stripped.lower() in ["na", "n/a", "none", "-", "null"]:
                rejected.append((row_number, raw_value))
                continue
                
            try:
                val = float(stripped)
                if math.isnan(val) or math.isinf(val):
                    rejected.append((row_number, raw_value))
                else:
                    valid_values.append(val)
            except ValueError:
                rejected.append((row_number, raw_value))
                
    return valid_values, rejected


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate mean of an empty list.")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate median of an empty list.")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    else:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""
    if not values:
        raise ValueError("Cannot calculate mode of an empty list.")
    
    counts = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1
        
    max_count = max(counts.values())
    
    # Per instructions: Document how you treat 'every value appears once'
    # If the maximum frequency is 1, it implies no single value is dominant or repeats.
    if max_count == 1:
        return []
        
    modes = [k for k, v in counts.items() if v == max_count]
    return sorted(modes)


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    n = len(values)
    if n == 0:
        raise ValueError("Cannot calculate variance of an empty list.")
    if sample and n == 1:
        raise ValueError("Sample variance requires at least 2 data points.")
        
    m = mean(values)
    squared_diff_sum = sum((x - m) ** 2 for x in values)
    
    denominator = (n - 1) if sample else n
    return squared_diff_sum / denominator


def std_dev(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample=sample))


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """
    Return (Q1, Q2, Q3). 
    
    QUARTILE METHOD SELECTION:
    We use linear percentile interpolation (Hazen's / inclusive rule), mapping the rank position
    at fractional indices via: index = p * (n - 1). This exact method maps directly to the 
    default configuration of numpy.percentile(..., method='linear') for parity checking.
    """
    if len(values) < 2:
        raise ValueError("Quartiles require at least 2 data points.")
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    def get_percentile(p: float) -> float:
        idx_float = p * (n - 1)
        idx_floor = int(math.floor(idx_float))
        idx_ceil = int(math.ceil(idx_float))
        
        if idx_floor == idx_ceil:
            return float(sorted_vals[idx_floor])
            
        weight = idx_float - idx_floor
        return sorted_vals[idx_floor] * (1.0 - weight) + sorted_vals[idx_ceil] * weight

    return get_percentile(0.25), get_percentile(0.50), get_percentile(0.75)


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    if not values:
        raise ValueError("Cannot detect outliers in an empty list.")
        
    if len(values) < 2:
        return {
            "iqr": 0.0,
            "lower_fence": float('-inf'),
            "upper_fence": float('inf'),
            "outliers": []
        }

        
    q1, q2, q3 = quartiles(values)
    iqr = q3 - q1
    lower_fence = q1 - (k * iqr)
    upper_fence = q3 + (k * iqr)
    
    outliers = [x for x in values if x < lower_fence or x > upper_fence]
    
    return {
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outliers": outliers
    }


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values: list[float], name: str = "values") -> None:
        self.values = values
        self.sorted_values = sorted(values)
        self.name = name
        self.rejected_records = []
        self._cached_summary = None

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        valid, rejected = load_column(path, column)
        instance = cls(valid, name=column)
        instance.rejected_records = rejected
        return instance

    def summary(self) -> dict:
        if self._cached_summary is not None:
            return self._cached_summary
            
        if not self.values:
            return {}
            
        q1, q2, q3 = quartiles(self.values)
        outlier_data = iqr_outliers(self.values)
        
        self._cached_summary = {
            "count": len(self.values),
            "mean": mean(self.values),
            "median": median(self.values),
            "mode": mode(self.values),
            "sample_variance": variance(self.values, sample=True) if len(self.values) > 1 else 0.0,
            "sample_std_dev": std_dev(self.values, sample=True) if len(self.values) > 1 else 0.0,
            "min": self.sorted_values[0] if self.values else 0.0,
            "max": self.sorted_values[-1] if self.values else 0.0,
            "q1": q1,
            "q3": q3,
            "iqr": outlier_data["iqr"],
            "lower_fence": outlier_data["lower_fence"],
            "upper_fence": outlier_data["upper_fence"],
            "outliers": outlier_data["outliers"]
        }
        return self._cached_summary

    def without_outliers(self) -> "StatisticsAnalyzer":
        if not self.values:
            return StatisticsAnalyzer([], name=f"{self.name}_cleaned")
            
        s = self.summary()
        lf, uf = s["lower_fence"], s["upper_fence"]
        cleaned_values = [x for x in self.values if lf <= x <= uf]
        
        instance = StatisticsAnalyzer(cleaned_values, name=f"{self.name}_cleaned")
        instance.rejected_records = self.rejected_records
        return instance

    def report(self) -> str:
        s = self.summary()
        if not s:
            return f"Dataset Profile '{self.name}' contains no valid rows."
            
        lines = [
            f"--- Analysis Summary: {self.name} ---",
            f"Total Valid Records : {s['count']}",
            f"Arithmetic Mean     : {s['mean']:.4f}",
            f"Median Value (Q2)   : {s['median']:.4f}",
            f"Calculated Mode     : {s['mode']}",
            f"Sample Variance     : {s['sample_variance']:.4f}",
            f"Sample Std Deviation: {s['sample_std_dev']:.4f}",
            f"Minimum Value       : {s['min']:.4f}",
            f"Maximum Value       : {s['max']:.4f}",
            f"First Quartile (Q1) : {s['q1']:.4f}",
            f"Third Quartile (Q3) : {s['q3']:.4f}",
            f"Interquartile Range : {s['iqr']:.4f}",
            f"Lower Fence Threshold: {s['lower_fence']:.4f}",
            f"Upper Fence Threshold: {s['upper_fence']:.4f}",
            f"Anomalies Outliers  : {len(s['outliers'])} items -> {s['outliers']}"
        ]
        return "\n".join(lines)


def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np
    
    s = analyzer.summary()
    if not s:
        return {}
        
    arr = np.array(analyzer.values)
    
    np_stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(np.percentile(arr, 25, method='linear')) if len(arr) >= 2 else s["q1"],
        "q3": float(np.percentile(arr, 75, method='linear')) if len(arr) >= 2 else s["q3"],
    }
    
    if len(arr) > 1:
        np_stats["sample_variance"] = float(np.var(arr, ddof=1))
        np_stats["sample_std_dev"] = float(np.std(arr, ddof=1))
    else:
        np_stats["sample_variance"] = 0.0
        np_stats["sample_std_dev"] = 0.0
        
    matches = {}
    for key, np_val in np_stats.items():
        matches[key] = math.isclose(s[key], np_val, rel_tol=1e-5)
        
    return matches


if __name__ == "__main__":
    import os
    
    # Quick sanity check for the presence of the data folder configuration
    if not DATA_PATH.exists():
        print(f"Error: Required sample file not found at path location: {DATA_PATH}")
        print("Please place the target agent_deposits.csv into the data/ subdirectory.")
        exit(1)

    os.makedirs("output", exist_ok=True)
    # 1. Initialize analyzer from input CSV
    raw_metrics = StatisticsAnalyzer.from_csv(DATA_PATH, "deposit_total")
    # 2. Filter outliers via the IQR fences
    cleaned_metrics = raw_metrics.without_outliers()
    # 3. Recompute validation against NumPy runtime matrices
    numpy_verification = compare_with_numpy(raw_metrics)
    # 4. Generate and write structured output/stats_report.txt summary document
    report_target_path = Path("output") / "stats_report.txt"
    with open(report_target_path, "w", encoding="utf-8") as out_file:
        out_file.write("=======================================================================\n")
        out_file.write("            WAKALA PLUS AGENT SYSTEM - STRATEGIC RISK REPORT           \n")
        out_file.write("=======================================================================\n\n")
        out_file.write("[SECTION 1: UNFILTERED ALL DATASET OBSERVATIONS]\n")
        out_file.write(raw_metrics.report() + "\n\n")
        out_file.write("-----------------------------------------------------------------------\n\n")
        out_file.write("[SECTION 2: CLEANED DATA PROFILE (OUTLIERS EXCISED)]\n")
        out_file.write(cleaned_metrics.report() + "\n\n")
        out_file.write("=======================================================================\n")
        out_file.write("               NUMPY PARITY VALIDATION ASSURANCE RESULTS               \n")
        out_file.write("=======================================================================\n")
        for stat_name, matches in numpy_verification.items():
            status_lbl = "MATCH" if matches else "MISMATCH"
            out_file.write(f"Parity Check Assertion -> {stat_name:<18}: {status_lbl}\n")
        out_file.write("\n=======================================================================\n")
        out_file.write("               RISK OPERATIONS TEAM BRIEFING STRATEGY                  \n")
        out_file.write("=======================================================================\n")
        out_file.write(
            "1. Low outliers dropping under the fence bounds highlight structural risks such\n"
            "   as system hardware drops or severe float shortages at rural sub-agent terminals.\n"
            "2. Extreme positive deposit anomalies track spikes requiring verification against ledger\n"
            "   streams to systematically rule out fraudulent cash-looping behaviors or application exploits.\n"
            "3. Eliminating these transactional outliers reduces metric standard deviation significantly,\n"
            "   isolating a clean operational baseline for forecasting regional capital requirements.\n"
        )
    print(f"Execution complete. Strategic report generated cleanly at: {report_target_path}")