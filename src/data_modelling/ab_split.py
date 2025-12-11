class ABGrouper:
    """
    Splits the dataset into Group A vs Group B for hypothesis testing.
    Example:
        Feature: Province
        Group A = Gauteng
        Group B = Western Cape
    """

    def __init__(self, df):
        self.df = df

    def create_groups(self, feature, group_a_value, group_b_value):
        df = self.df.copy()

        group_a = df[df[feature] == group_a_value]
        group_b = df[df[feature] == group_b_value]

        return group_a, group_b
