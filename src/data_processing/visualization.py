import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Ensure TransactionMonth datetime
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(
                self.df['TransactionMonth'], errors='coerce'
            )

        # FIX POSTAL CODE PROBLEM
        if 'PostalCode' in self.df.columns:
            self.df['PostalCode'] = (
                self.df['PostalCode']
                .astype(str)
                .str.strip()
                .replace(['', 'nan', 'NaN', 'None'], pd.NA)
            )

            # convert to numeric only if possible
            self.df['PostalCode'] = pd.to_numeric(
                self.df['PostalCode'], errors='coerce'
            )

            # drop rows where postal is missing
            self.df = self.df.dropna(subset=['PostalCode'])

            # convert to int
            self.df['PostalCode'] = self.df['PostalCode'].astype(int)

    # ---------------------------------------------------------
    # 1. Premium vs Claims scatter
    # ---------------------------------------------------------
    def plot_premium_vs_claims(self):
        plt.figure(figsize=(10,5))
        sns.scatterplot(
            data=self.df,
            x='TotalPremium',
            y='TotalClaims',
            hue='CoverType',
            alpha=0.6
        )
        plt.title("Total Premium vs Total Claims")
        plt.show()

    # ---------------------------------------------------------
    # 2. CoverCategory by Province
    # ---------------------------------------------------------
    def plot_covercategory_by_province(self):
        tab = pd.crosstab(self.df['Province'], self.df['CoverCategory'])
        tab = tab.div(tab.sum(axis=1), axis=0)

        tab.plot(kind='bar', stacked=True, figsize=(12,6))
        plt.title("Cover Category Distribution by Province")
        plt.show()

    # ---------------------------------------------------------
    # 3. Monthly Premium Trends for TOP Postal Codes ONLY
    # ---------------------------------------------------------
    def plot_monthly_premium_top_postal(self, top_n=5):
        # group monthly premium per postal code
        gp = (
            self.df
            .groupby(['TransactionMonth', 'PostalCode'])['TotalPremium']
            .sum()
            .reset_index()
        )

        # get top postal codes by total premium
        top_codes = (
            gp.groupby('PostalCode')['TotalPremium']
            .sum()
            .nlargest(top_n)
            .index
        )

        # filter
        filtered = gp[gp['PostalCode'].isin(top_codes)]

        plt.figure(figsize=(12,6))
        sns.lineplot(
            data=filtered,
            x='TransactionMonth',
            y='TotalPremium',
            hue='PostalCode',
            marker='o'
        )
        plt.title(f"Monthly Premium Trends for Top {top_n} Postal Codes")
        plt.xticks(rotation=45)
        plt.show()
