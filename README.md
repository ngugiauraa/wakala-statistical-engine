# Custom Descriptive Statistics Engine

A production-grade, plain-Python descriptive statistics engine engineered entirely from first principles. This toolkit is built to parse, clean, and analyze high-volatility financial streaming records, providing a granular statistical analysis of transactional flows without relying on black-box analytics libraries. 

The core math layer is written using pure Python data structures and algorithms, using NumPy strictly as an external assertion layer within the test bench to guarantee mathematical precision.

## Core Features
* **Zero-Dependency Math Pipeline:** Custom implementations of central tendency (mean, median, multi-tie mode) and statistical dispersion (sample/population variance, standard deviation).
* **Automated Data Hygiene & Auditing:** Real-time data pipeline parser that catches and strips out incomplete records, malformed numeric string literals, or infinite values while preserving a distinct row-by-row audit trail.
* **Hazen/Inclusive Quantile Framework:** High-precision quartile estimation using linear interpolation to isolate distribution percentiles accurately.
* **IQR-Based Anomaly Detection:** Interquartile Range (IQR) fence profiling to isolate, extract, and report operational outliers dynamically.
* **Self-Validating Architecture:** Built-in validation suite that hooks into external vector environments to double-check precision down to a specific relative tolerance.

---

## Architectural Choices & Mathematical Strategy

### 1. Granular Variance Estimation
The analytics layer defaults to **Sample Variance** (n - 1 degrees of freedom) when assessing transaction distribution vectors. This approach applies Bessel’s correction to eliminate down-biased variance estimations inherent in localized batch slices, making it highly reliable for risk forecasting on rolling infrastructure data.

### 2. Linear Interpolation Percentile Mapping
To ensure perfect parity with standard data science runtimes, percentiles are mapped using the linear interpolation inclusive framework (Hyndman-Fan Type 7 criteria). The rank placement is determined by:
\[\text{position} = p \times (n - 1)\]
When the computed vector position falls on a fractional boundary, the value is interpolated using the linear distances between neighboring sorted indices. This design guarantees identical precision when verified against automated matrix operations.

### 3. Non-Destructive Data Hygiene
The file parser enforces strict type hygiene without resorting to destructive data imputation or arbitrary value guessing. Missing segments or parsing errors are extracted into a separate rejection stream with their exact source metadata. This isolates corrupted records immediately without compromising the integrity of the downstream aggregation layer.

---

## Technical Overview & Quick Start

Ensure you have a local installation of Python 3.12+ configured.

### 1. Initialize the Isolated Workspace
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows PowerShell use: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Statistical Analysis Pipeline
Execute the root module runner script to ingest raw dataset components, strip statistical anomalies, and dump a comprehensive data summary profile:
```bash
python starter.py
```
*Generated file output path:* `output/stats_report.txt`

### 3. Execute the Automated Test Bench
Run the regression test suite across the verified mathematical routines and data ingestion layers:
```bash
python -m unittest tests/test_stats_toolkit.py -v
```

---

## Workspace Layout
```wakala-statistical-engine
├── starter.py               # Core descriptive statistics algorithms & Analyzer class
├── tests/
│   └── test_stats_toolkit.py # Comprehensive unit test suite with 9 passing checks
├── data/
│   └── agent_deposits.csv    # Sample high-volatility transaction records
├── output/
│   └── stats_report.txt     # Generated comparative statistical reports
├── requirements.txt         # Verification environment tracking layers (NumPy)
└── README.md                # System documentation
```

Built by: [Jesse Ngugi](https://ngugi.pages.dev)
