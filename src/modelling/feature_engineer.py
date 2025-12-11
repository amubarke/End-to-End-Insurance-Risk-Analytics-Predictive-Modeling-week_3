import pandas as pd

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def engineer(self) -> pd.DataFrame:
        df = self.df.copy()

        # Vehicle age
        if "RegistrationYear" in df.columns:
            df["VehicleAge"] = 2025 - df["RegistrationYear"]

        # Claim indicator (if not already present)
        if "claim_indicator" not in df.columns and "TotalClaims" in df.columns:
            df["claim_indicator"] = (df["TotalClaims"] > 0).astype(int)

        # You can add more engineered features here

        return df
