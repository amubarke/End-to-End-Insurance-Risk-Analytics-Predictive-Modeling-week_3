class ReportGenerator:
    """
    Produces readable interpretation of hypothesis results.
    """

    def build_conclusion(self, p_value, alpha=0.05, h0_text=""):
        if p_value < alpha:
            return f"Reject H₀: {h0_text} → There IS a significant difference."
        else:
            return f"Fail to Reject H₀: {h0_text} → No significant difference detected."
