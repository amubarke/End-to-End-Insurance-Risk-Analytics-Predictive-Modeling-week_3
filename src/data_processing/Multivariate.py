import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MultivariateAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    # ---- Reduce over-dense categories (PostalCode, Model, etc.) ----
    def _reduce_categories(self, series, top_n=20):
        series = series.astype(str)
        freq = series.value_counts()
        top_cats = freq.head(top_n).index
        return series.where(series.isin(top_cats), other="Other")

    # ---- Compute monthly change for premium & claims ----------------
    def compute_monthly_changes(self):
        if "TransactionMonth" not in self.df.columns:
            raise ValueError("TransactionMonth column missing")

        df_sorted = self.df.sort_values("TransactionMonth")

        self.df["Premium_Change"] = df_sorted["TotalPremium"].diff()
        self.df["Claims_Change"] = df_sorted["TotalClaims"].diff()

    # ---- Scatter plot change in Premium vs Claims ------------------
    def scatter_premium_vs_claims(self, category_col=None, top_n=20):
        df = self.df.copy()

        # Optional grouping
        if category_col:
            df[category_col] = self._reduce_categories(df[category_col], top_n)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="Premium_Change",
            y="Claims_Change",
            hue=category_col if category_col else None,
            alpha=0.6
        )
        plt.title("Premium Change vs Claims Change")
        plt.tight_layout()
        plt.show()

    # ---- Correlation Matrix of all numeric vars --------------------
    def correlation_matrix(self):
        numeric_df = self.df.select_dtypes(include=["int64", "float64", "Int64", "Float64"])

        plt.figure(figsize=(14, 10))
        sns.heatmap(numeric_df.corr(), annot=False, cmap="Blues")
        plt.title("Correlation Matrix of Numerical Features")
        plt.tight_layout()
        plt.show()

    # ---- Full function to run multivariate EDA ---------------------
    def run_all(self):
        print("📌 Computing monthly changes...")
        self.compute_monthly_changes()

        print("📌 Correlation matrix...")
        self.correlation_matrix()

        print("📌 Scatter: Premium Change vs Claims Change by PostalCode...")
        self.scatter_premium_vs_claims("PostalCode", top_n=15)

        print("📌 Scatter: Premium Change vs Claims Change by Model...")
        self.scatter_premium_vs_claims("Model", top_n=15)
