import pandas as pd

class DataSummary:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a dataframe.
        """
        self.df = df.copy()

    def info(self):
        """
        Returns dataframe info: data types, non-null counts, memory usage.
        """
        buffer = []
        self.df.info(buf=buffer)
        return buffer  # or just use print(self.df.info())

    def describe(self):
        """
        Returns summary statistics for numerical columns.
        """
        return self.df.describe().T

    def null_values(self):
        """
        Returns missing value count and percentage per column.
        """
        null_count = self.df.isnull().sum()
        null_percent = (null_count / len(self.df)) * 100
        return pd.DataFrame({
            "null_count": null_count,
            "null_percent": null_percent
        })

    def unique_values(self):
        """
        Returns count of unique values per column.
        """
        return self.df.nunique().sort_values(ascending=False)

    def top_frequent(self, n=5):
        """
        Returns top n frequent values per categorical column.
        """
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        top_vals = {}
        for col in cat_cols:
            top_vals[col] = self.df[col].value_counts().head(n).to_dict()
        return top_vals
