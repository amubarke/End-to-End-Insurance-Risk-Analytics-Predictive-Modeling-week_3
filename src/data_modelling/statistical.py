from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from statsmodels.stats.proportion import proportions_ztest

class HypothesisTester:
    def __init__(self, df):
        self.df = df.copy()

    # ---------- Chi-square test ----------
    def chi_square_test(self, feature, kpi):
        """Chi-square test for categorical variables (frequency)."""
        contingency = pd.crosstab(self.df[feature], self.df[kpi])
        chi2, p, dof, expected = chi2_contingency(contingency)
        return {"p_value": p, "chi2": chi2, "table": contingency}

    # ---------- Two-proportion z-test ----------
    def proportion_test(self, group_a, group_b, kpi_col="has_claim"):
        """Z-test for two proportions."""
        count = [group_a[kpi_col].sum(), group_b[kpi_col].sum()]
        nobs = [len(group_a), len(group_b)]
        stat, p = proportions_ztest(count, nobs)
        return {"p_value": p, "statistic": stat}

    # ---------- t-test ----------
    def t_test(self, group_a, group_b, column):
        """t-test for means (numeric KPI)."""
        stat, p = ttest_ind(group_a[column].dropna(), group_b[column].dropna())
        return {"p_value": p, "statistic": stat}

    # ---------- ANOVA ----------
    def anova_test(self, feature, numeric_col):
        """ANOVA for numeric KPI across >2 groups."""
        groups = [
            group[numeric_col].dropna()
            for name, group in self.df.groupby(feature)
        ]
        stat, p = f_oneway(*groups)
        return {"p_value": p, "f_statistic": stat}
