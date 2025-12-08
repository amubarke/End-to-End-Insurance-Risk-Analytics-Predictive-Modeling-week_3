import pandas as pd
import matplotlib.pyplot as plt

class GeographicAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def plot_top10(self, column_name: str, value_name: str = None):
        """
        Plot the TOP 10 values for a categorical geographic column.
        Good for: PostalCode, CoverType, make, etc.
        """
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found.")

        # Count values
        top10 = self.df[column_name].value_counts().head(10)

        plt.figure(figsize=(12, 6))
        top10.plot(kind="bar", color="skyblue")

        plt.title(f"Top 10 {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Count")

        plt.xticks(rotation=45, ha="right")

        # Adjust layout without warnings
        plt.subplots_adjust(bottom=0.3, top=0.9)

        plt.show()

    def plot_geo_trend(self, geo_col: str, num_col: str):
        """
        Plot average numerical value per geographic group.
        Good for: TotalPremium by PostalCode, SumInsured by Province, etc.
        """
        if geo_col not in self.df.columns:
            raise ValueError(f"'{geo_col}' not found in dataset.")
        if num_col not in self.df.columns:
            raise ValueError(f"'{num_col}' not found in dataset.")

        # Group and take top 10 only
        grouped = (
            self.df.groupby(geo_col)[num_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        plt.figure(figsize=(12, 6))
        grouped.plot(kind="bar", color="salmon")

        plt.title(f"Top 10 {geo_col}: Avg {num_col}")
        plt.xlabel(geo_col)
        plt.ylabel(f"Avg {num_col}")

        plt.xticks(rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.3, top=0.9)

        plt.show()

    def compare_categories(self, cat_col: str, num_col: str):
        """
        Compare numerical column across top 10 categories.
        Example: CoverType vs TotalPremium
        """
        if cat_col not in self.df.columns:
            raise ValueError(f"'{cat_col}' not found.")
        if num_col not in self.df.columns:
            raise ValueError(f"'{num_col}' not found.")

        top10 = (
            self.df.groupby(cat_col)[num_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        plt.figure(figsize=(12, 6))
        top10.plot(kind="bar", color="lightgreen")

        plt.title(f"Top 10 {cat_col} by Avg {num_col}")
        plt.xlabel(cat_col)
        plt.ylabel(f"Avg {num_col}")

        plt.xticks(rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.3, top=0.9)

        plt.show()
