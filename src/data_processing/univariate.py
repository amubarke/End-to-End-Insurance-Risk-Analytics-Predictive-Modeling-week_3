import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalyzer:
    def __init__(self, df):
        self.df = df

    # ---- Reduce categories so postal code / model do NOT look dense -------
    def _reduce_categories(self, series, top_n=10):
        # Convert to string to avoid Categorical assignment errors
        series = series.astype(str)

        freq = series.value_counts()
        top_categories = freq.head(top_n).index

        return series.where(series.isin(top_categories), other="Other")

    # ---- Plot ALL numerical columns ----------------------------------------
    def plot_all_numerical(self):
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns

        for col in numeric_cols:
            plt.figure(figsize=(10,5))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    # ---- Plot ALL categorical columns --------------------------------------
    def plot_all_categorical(self, top_n=10):
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            reduced = self._reduce_categories(self.df[col], top_n)

            plt.figure(figsize=(12,6))
            sns.countplot(y=reduced, order=reduced.value_counts().index)
            plt.title(f"Top {top_n} categories of {col}")
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()

    # ---- Combined function --------------------------------------------------
    def plot_all(self, top_n=10):
        print("📌 Plotting ALL numerical columns...")
        self.plot_all_numerical()

        print("📌 Plotting ALL categorical columns...")
        self.plot_all_categorical(top_n)
