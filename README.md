# MissingnessMatters

**MissingnessMatters** is a research-grade toolkit designed to audit, diagnose, and handle missing data in machine learning datasets. Using the Pima Indians Diabetes Dataset as a case study, this project demonstrates how to identify "Hidden Nulls," statistically differentiate between missingness mechanisms (MCAR vs. MAR), and benchmark advanced imputation strategies like MICE against baselines.

## 🚀 Key Features

*   **⚠️ Hidden Null Exposure**: Automatically detects and converts biologically impossible zeros (e.g., 0 BMI, 0 Glucose) into true `NaN` values.
*   **🔍 Statistical Auditing**:
    *   **Visual Audit**: Uses `missingno` to visualize missingness matrices and nullity correlations.
    *   **Mechanism Tests**: Implements **Welch's T-test** to statistically confirm missingness mechanisms (e.g., checking if Age influences Insulin missingness).
*   **🤖 Imputation Benchmarking**: Compare multiple imputation techniques:
    *   **Baseline**: Mean/Median Imputation.
    *   **Local**: KNN Imputation.
    *   **Global**: MICE (Multivariate Imputation by Chained Equations).
*   **📊 Distribution Shift Analysis**: Quantify how different imputation methods distort the original data distribution using **Kolmogorov-Smirnov (KS) tests** and KDE plots.

## 📂 Project Structure

```text
MissingnessMatters/
├── data/
│   ├── raw/                 # Original dataset (Pima Indians Diabetes)
│   └── processed/           # Processed datasets (Exposed Nulls, MICE imputed)
├── notebooks/
│   ├── mechanism_audit.ipynb       # Visual exploration of missing data patterns
│   └── imputation_benchmarks.ipynb # Implementation & comparison of imputation strategies
├── src/
│   ├── data_loader.py       # Pipeline to load raw data and expose hidden missingness
│   └── statistical_test.py  # Statistical suite for MCAR/MAR hypothesis testing
├── .venv/                   # Python Virtual Environment
└── README.md                # Project Documentation
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/MissingnessMatters.git
    cd MissingnessMatters
    ```

2.  **Set up the environment**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate  # On Windows
    # source .venv/bin/activate # On macOS/Linux
    ```

3.  **Install dependencies**:
    ```bash
    pip install pandas numpy scipy matplotlib seaborn missingno scikit-learn
    ```

## 🚦 Usage Guide

### 1. Data Loading & Preprocessing
Run the data loader to ingest the raw CSV and convert invalid zeros to NaNs.
```bash
python src/data_loader.py
```
*Output: Saves processed data to `data/processed/pima_exposed.csv`.*

### 2. Statistical Audit
Run the statistical test suite to check for Missing at Random (MAR) evidence.
```bash
python src/statistical_test.py
```
*Output: Prints results of T-tests checking if observed variables (like Age/BMI) correlate with the missingness of others (like Insulin).*

### 3. Notebook Exploration
*   **`notebooks/mechanism_audit.ipynb`**: Open this to visualize the missing data patterns.
*   **`notebooks/imputation_benchmarks.ipynb`**: Run this to generate imputed datasets and view comparison plots.

## 🧠 Methodology

This project follows a rigorous statistical framework:

1.  **Detection**: "Zero" is not always "Zero". We first validate logical consistency.
2.  **Diagnosis**:
    *   **MCAR (Missing Completely At Random)**: proven if $P(Missing|Data) = P(Missing)$.
    *   **MAR (Missing At Random)**: proven if $P(Missing|Y_{obs}) \neq P(Missing)$.
3.  **Treatment**: We prefer **MICE** over simple imputation because it preserves the relationships between variables, minimizing bias in downstream machine learning models.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
