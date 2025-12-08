import pandas as pd
import matplotlib.pyplot as plt

class FinancialAnalyzer:
    def __init__(self, df):
        self.df = df

    def plot_distributions(self, columns):
        """
        Plot histograms for financial variables
        """
        for col in columns:
            plt.figure(figsize=(7,5))
            self.df[col].dropna().hist(bins=40)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(False)
            plt.show()

    def plot_outliers(self, columns):
        """
        Create vertical box plots for outlier detection
        """
        for col in columns:
            plt.figure(figsize=(5,7))
            plt.boxplot(self.df[col].dropna(), vert=True)
            plt.title(f"Outlier Detection: {col}")
            plt.ylabel(col)
            plt.grid(False)
            plt.show()
