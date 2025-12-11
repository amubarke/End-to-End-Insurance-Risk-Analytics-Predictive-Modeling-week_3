class PremiumPricingEngine:
    """
    Premium = P(Claim) * Expected Severity + Expenses + Profit
    """

    def __init__(self, severity_model, probability_model):
        self.severity_model = severity_model
        self.probability_model = probability_model

    def calculate_premium(self, prob_claim, severity, expense_loading=0.10, profit_margin=0.15):
        """
        expense_loading: business costs (10%)
        profit_margin: profit buffer (15%)
        """
        base_premium = prob_claim * severity
        return base_premium * (1 + expense_loading + profit_margin)
