import pandas as pd

class DescriptiveAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with cleaned dataframe.
        """
        self.df = df
        self.numeric_df = df.select_dtypes(include=['float64', 'int64', 'Int64'])

    def basic_descriptive_stats(self):
        """
        Returns descriptive statistics for numerical columns.
        """
        return self.numeric_df.describe()

    def variability_metrics(self):
        """
        Returns a dataframe with variability statistics such as:
        std, variance, range, IQR, etc.
        """
        return pd.DataFrame({
            "mean": self.numeric_df.mean(),
            "std_dev": self.numeric_df.std(),
            "variance": self.numeric_df.var(),
            "min": self.numeric_df.min(),
            "max": self.numeric_df.max(),
            "range": self.numeric_df.max() - self.numeric_df.min(),
            "Q1": self.numeric_df.quantile(0.25),
            "median": self.numeric_df.median(),
            "Q3": self.numeric_df.quantile(0.75),
            "IQR": self.numeric_df.quantile(0.75) - self.numeric_df.quantile(0.25)
        })

    def summarize(self):
        """
        Prints both descriptive statistics and variability.
        """
        print("\n📌 DESCRIPTIVE STATISTICS\n")
        print(self.basic_descriptive_stats())

        print("\n📌 VARIABILITY METRICS\n")
        print(self.variability_metrics())



