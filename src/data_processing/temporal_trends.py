import pandas as pd
import matplotlib.pyplot as plt

class TemporalClaimsAnalyzer:
    def __init__(self, df, date_column):
        self.df = df.copy()
        self.date_column = date_column
        self.prepare_dates()

    def prepare_dates(self):
        """Convert date column to datetime & create Year-Month"""
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors='coerce')
        self.df.dropna(subset=[self.date_column], inplace=True)
        self.df['YearMonth'] = self.df[self.date_column].dt.to_period('M')

    def compute_monthly_metrics(self):
        """Compute claim frequency and severity trends"""
        monthly = self.df.groupby('YearMonth').agg(
            ClaimFrequency=('TotalClaims', lambda x: (x > 0).sum()),
            TotalClaimAmount=('TotalClaims', 'sum')
        )
        monthly['ClaimSeverity'] = monthly['TotalClaimAmount'] / monthly['ClaimFrequency']
        monthly.replace([float('inf'), -float('inf')], 0, inplace=True)
        return monthly

    def plot_trends(self):
        """Plot frequency and severity trends"""
        monthly = self.compute_monthly_metrics()

        #  Claim Frequency Trend
        plt.figure(figsize=(10,5))
        plt.plot(monthly.index.astype(str), monthly['ClaimFrequency'])
        plt.xticks(rotation=45)
        plt.title("📈 Monthly Claim Frequency Trend")
        plt.xlabel("Month")
        plt.ylabel("Number of Claims")
        plt.grid(False)
        plt.show()

        # Claim Severity Trend
        plt.figure(figsize=(10,5))
        plt.plot(monthly.index.astype(str), monthly['ClaimSeverity'])
        plt.xticks(rotation=45)
        plt.title("📉 Monthly Claim Severity Trend")
        plt.xlabel("Month")
        plt.ylabel("Average Claim Amount")
        plt.grid(False)
        plt.show()

        # Combined Plot
        plt.figure(figsize=(12,6))
        plt.plot(monthly.index.astype(str), monthly['ClaimFrequency'], label="Claim Frequency")
        plt.plot(monthly.index.astype(str), monthly['ClaimSeverity'], label="Claim Severity")
        plt.xticks(rotation=45)
        plt.title("📊 Frequency & Severity Trends Over Time")
        plt.xlabel("Month")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(False)
        plt.show()
