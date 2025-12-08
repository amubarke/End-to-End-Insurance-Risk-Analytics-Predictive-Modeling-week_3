import pandas as pd
import matplotlib.pyplot as plt

class VehicleClaimAnalyzer:
    def __init__(self, df, make_col="make", model_col="Model", claim_col="TotalClaims"):
        self.df = df.copy()
        self.make_col = make_col
        self.model_col = model_col
        self.claim_col = claim_col

        self._prepare()

    def _prepare(self):
        """ Basic cleaning for analysis """
        self.df[self.claim_col] = pd.to_numeric(self.df[self.claim_col], errors='coerce').fillna(0)
        self.df[self.make_col] = self.df[self.make_col].astype(str)
        self.df[self.model_col] = self.df[self.model_col].astype(str)

        self.df["Make_Model"] = self.df[self.make_col] + " " + self.df[self.model_col]

    def compute_claim_totals(self):
        """ Sum total claim amounts per vehicle make/model """
        grouped = (
            self.df.groupby("Make_Model")[self.claim_col]
            .sum()
            .sort_values(ascending=False)
        )
        return grouped

    def plot_top_bottom(self, top_n=10):
        """ Plot top & bottom vehicle claim totals """
        totals = self.compute_claim_totals()

        top_models = totals.head(top_n)
        bottom_models = totals.tail(top_n)

        # --- Top 10 Highest Claim Models ---
        plt.figure(figsize=(10, 6))
        top_models.plot(kind="bar")
        plt.title("🚗 Top Vehicle Models With Highest Claim Amounts")
        plt.xlabel("Vehicle Make + Model")
        plt.ylabel("Total Claim Amount")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # --- Bottom 10 Lowest Claim Models ---
        plt.figure(figsize=(10, 6))
        bottom_models.plot(kind="bar")
        plt.title("🚙 Vehicle Models With Lowest Claim Amounts")
        plt.xlabel("Vehicle Make + Model")
        plt.ylabel("Total Claim Amount")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def summary(self, top_n=10):
        totals = self.compute_claim_totals()
        return {
            "Top Models": totals.head(top_n),
            "Bottom Models": totals.tail(top_n)
        }
