class ResultReporter:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def interpret(self, p_value, null_hypothesis):
        """
        Interpret statistical test results.
        """
        if p_value < self.alpha:
            return f"Reject H₀: {null_hypothesis}"
        else:
            return f"Fail to Reject H₀: {null_hypothesis}"

    def print_result(self, test_name, p_value, null_hypothesis):
        decision = self.interpret(p_value, null_hypothesis)
        print(f"--- {test_name} ---")
        print(f"p-value: {p_value:.6f}")
        print(f"Decision: {decision}\n")
