import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal, mannwhitneyu

class HypothesisTester:
    """
    Performs all hypothesis tests for  project:
    1. Provincial risk differences
    2. ZIP code risk differences
    3. ZIP code margin (profit) differences
    4. Gender-based risk differences
    """

    def __init__(self, df):
        self.df = df.copy()

    # -------------------------
    # Utility decision helper
    # -------------------------
    def decision(self, p_value, alpha=0.05):
        return "Reject H₀" if p_value < alpha else "Fail to Reject H₀"

    # -------------------------------------------------------------
    # 1. Risk Differences Across Provinces
    # -------------------------------------------------------------
    def test_province_risk_difference(self, target_col="TotalClaims"):
        df = self.df.copy()

        # Group by province
        groups = [
            group[target_col].dropna().values
            for _, group in df.groupby("Province")
        ]

        stat, p_value = f_oneway(*groups)

        return {
            "test": "Province Risk Difference (ANOVA)",
            "p_value": p_value,
            "decision": self.decision(p_value),
            "target_column": target_col
        }

    # -------------------------------------------------------------
    # 2. Risk Differences Between ZIP Codes
    # -------------------------------------------------------------
    def test_zip_risk_difference(self, target_col="TotalClaims", top_n=10):
        df = self.df.copy()

        top_zip = df["PostalCode"].value_counts().nlargest(top_n).index
        df = df[df["PostalCode"].isin(top_zip)]

        groups = [
            group[target_col].dropna().values
            for _, group in df.groupby("PostalCode")
        ]

        stat, p_value = f_oneway(*groups)

        return {
            "test": "ZIP Code Risk Difference (ANOVA)",
            "p_value": p_value,
            "decision": self.decision(p_value),
            "target_column": target_col
        }

    # -------------------------------------------------------------
    # 3. Margin (Profit) Differences Between Zip Codes
    # -------------------------------------------------------------
    def test_zip_margin_difference(self, top_n=10):
        df = self.df.copy()

        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

        top_zip = df["PostalCode"].value_counts().nlargest(top_n).index
        df = df[df["PostalCode"].isin(top_zip)]

        groups = [
            group["Margin"].dropna().values
            for _, group in df.groupby("PostalCode")
        ]

        stat, p_value = f_oneway(*groups)

        return {
            "test": "ZIP Code Margin Difference (ANOVA)",
            "p_value": p_value,
            "decision": self.decision(p_value)
        }

    # -------------------------------------------------------------
    # 4. Gender-Based Risk Differences
    # -------------------------------------------------------------
    def test_gender_risk_difference(self, target_col="TotalClaims"):
        df = self.df.copy()

        male = df[df["Gender"] == "Male"][target_col].dropna()
        female = df[df["Gender"] == "Female"][target_col].dropna()

        stat, p_value = mannwhitneyu(male, female)

        return {
            "test": "Gender Risk Difference (Mann–Whitney U)",
            "p_value": p_value,
            "decision": self.decision(p_value),
            "target_column": target_col
        }
