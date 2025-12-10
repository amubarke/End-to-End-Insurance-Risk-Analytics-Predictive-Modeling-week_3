import pandas as pd
import numpy as np

class KPISelector:
    def __init__(self, df):
        self.df = df.copy()

    def compute_kpis(self):
        """Create KPIs: claim frequency, claim severity, margin."""
        df = self.df

        # Claim Frequency
        df["has_claim"] = df["TotalClaims"].apply(lambda x: 1 if x > 0 else 0)

        # Claim Severity (only where claim amount exists)
        df["claim_severity"] = np.where(
            (df["TotalClaims"] > 0) & (df["TotalClaimsAmount"] > 0),
            df["TotalClaimsAmount"] / df["TotalClaims"],
            np.nan
        )

        # Margin
        df["margin"] = df["TotalPremium"] - df["TotalClaimsAmount"].fillna(0)

        return df[["has_claim", "claim_severity", "margin"]]
