class DataSegmenter:
    def __init__(self, df):
        self.df = df.copy()

    def split_ab(self, column, group_a_value, group_b_value):
        """
        Split data into A and B groups based on a specific feature.
        Ensure both groups exist.
        """
        group_a = self.df[self.df[column] == group_a_value]
        group_b = self.df[self.df[column] == group_b_value]

        if len(group_a) == 0 or len(group_b) == 0:
            raise ValueError("One of the groups is empty!")

        return group_a, group_b
