import pandas as pd
import numpy as np

class KPISelector:
    """
    Computes KPIs needed for hypothesis testing:
    - Claim Frequency
    - Claim Severity
    - Margin
    """

    def __init__(self, df):
        self.df = df

    def compute_kpis(self):
        df = self.df.copy()

        # Claim Frequency: Did a claim occur?
        df["claim_frequency"] = df["TotalClaims"].apply(lambda x: 1 if x > 0 else 0)

        # Claim Severity: Average claim amount (best available proxy)
        df["claim_severity"] = df["TotalClaims"].replace(0, np.nan)

        # Margin = Premium - Claims
        df["margin"] = df["TotalPremium"] - df["TotalClaims"]

        return df
