import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierDetector:
    def __init__(self, df: pd.DataFrame, columns=None):
        """
        Initialize with dataframe and list of numeric columns to check for outliers.
        """
        self.df = df.copy()
        self.columns = columns or []

    def plot_boxplots(self):
        """
        Plots vertical boxplots for selected numeric columns to detect outliers.
        """
        if not self.columns:
            print("No columns specified for outlier detection.")
            return
        
        for col in self.columns:
            plt.figure(figsize=(6, 6))
            sns.boxplot(y=self.df[col], color='skyblue')
            plt.title(f"Outlier Detection: {col}")
            plt.ylabel(col)
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.show()
    
    def get_outliers(self):
        """
        Returns a dictionary of outliers for each column using IQR method.
        """
        outlier_dict = {}
        for col in self.columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            outlier_dict[col] = outliers
        return outlier_dict
