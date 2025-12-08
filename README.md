# Week 3 - End-to-End Insurance Risk Analytics: Interim Report

## Overview
This notebook covers **Week 3** of the End-to-End Insurance Risk Analytics project. The focus is on **data cleaning, exploratory data analysis (EDA), and preliminary insights** from the insurance portfolio.

**Objectives:**
1. Clean the raw insurance dataset.
2. Perform descriptive statistics, univariate and multivariate analysis.
3. Visualize financial distributions, outliers, and temporal trends.
4. Analyze vehicle-level claims and loss ratios.
5. Prepare insights for predictive modeling in subsequent weeks.

---

## Project Structure

project_root/
│
├── data/
│ ├── MachineLearningRating_v3.csv # Original raw dataset
│ ├── cleaned_data.csv # Cleaned dataset after preprocessing
│ └── Week3_Interim_Report.pdf # Generated PDF report
│
├── notebooks/
│ └── Week3_EDA.ipynb # This notebook
│
├── src/
│ └── data_processing/
│ ├── data_loader.py # Functions to load CSV/TXT files
│ ├── data_cleaner.py # Class for cleaning and preprocessing
│ └── data_summary.py # Class for descriptive statistics and EDA
│
└── README.md


---

## Data Cleaning

**Steps performed:**
- Removed empty or irrelevant columns (`NumberOfVehiclesInFleet`).
- Converted date columns (`TransactionMonth`, `VehicleIntroDate`) to datetime.
- Converted financial columns (`CapitalOutstanding`, `TermFrequency`) to numeric types.
- Converted categorical columns (e.g., `PolicyID`, `VehicleType`, `Province`, `CoverType`) to `category`.
- Filled missing values:
  - Numerical: median or zero
  - Categorical: mode or `"Unknown"`
- Removed duplicate rows.

**Result:** `df_cleaned` ready for analysis.

---

## EDA Highlights

### 1️⃣ Descriptive Statistics
- Key financial variables analyzed: `TotalPremium`, `TotalClaims`, `CustomValueEstimate`, etc.
- Outliers detected in `TotalClaims` and `CustomValueEstimate`.

### 2️⃣ Univariate Analysis
- Histograms and bar charts for numerical and categorical features.

### 3️⃣ Bivariate / Multivariate Analysis
- Loss Ratio analyzed by `Province`, `VehicleType`, and `Gender`.
- Vehicle-level claims analyzed for highest and lowest claim amounts.

### 4️⃣ Temporal Analysis
- Claim frequency and severity trends over 18 months.
- Insights into seasonality and portfolio exposure.

---

## Usage Examples

**Load Cleaned Data**
```python
from src.data_processing.data_loader import load_data

data_path = "../data/cleaned_data.csv"
df_cleaned = load_data(data_path)

Generate Loss Ratio Plots

from src.data_processing.data_summary import LossRatioVisualizer

viz = LossRatioVisualizer(df_cleaned)
viz.plot_loss_by_province()
viz.plot_loss_by_vehicle_type()
viz.plot_loss_by_gender()

Financial Distribution and Outlier Analysis

from src.data_processing.data_summary import FinancialAnalyzer

analyzer = FinancialAnalyzer(df_cleaned)
analyzer.plot_distributions(["TotalClaims", "CustomValueEstimate"])
analyzer.plot_outliers(["TotalClaims", "CustomValueEstimate"])

Vehicle Claim Analysis

from src.data_processing.data_summary import VehicleClaimAnalyzer

vehicle_analyzer = VehicleClaimAnalyzer(df_cleaned, make_col="make", model_col="Model", claim_col="TotalClaims")
vehicle_analyzer.plot_top_bottom()

Temporal Trend Analysis

from src.data_processing.data_summary import TemporalClaimsAnalyzer

temporal_analyzer = TemporalClaimsAnalyzer(df_cleaned, date_column="TransactionMonth")
temporal_analyzer.plot_trends()

Output

cleaned_data.csv → Preprocessed dataset for analysis and modeling.

Notes

Missing values in CustomValueEstimate are significant (~78%), consider handling in predictive models.

Outliers in TotalClaims and CustomValueEstimate may affect model performance; consider transformation or capping.

Postal code has been retained but may be aggregated for reporting to reduce plot density.

Next Steps

Feature engineering for modeling: frequency, severity, and vehicle risk features.

Predictive modeling for claims frequency and severity.

Integrate findings into Week 4 risk assessment and forecasting.