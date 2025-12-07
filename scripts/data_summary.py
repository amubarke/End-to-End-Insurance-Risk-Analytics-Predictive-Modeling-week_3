import pandas as pd

class RawDataSummarizer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a raw dataframe (no cleaning done)
        """
        self.df = df

    def numeric_stats(self):
        """Compute descriptive statistics and variability for all numeric columns"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        stats = self.df[numeric_cols].describe().T  # transpose for readability
        stats['median'] = self.df[numeric_cols].median()
        stats['range'] = stats['max'] - stats['min']
        stats['variance'] = self.df[numeric_cols].var()
        stats['std_dev'] = self.df[numeric_cols].std()
        print("=== Numeric Descriptive Statistics ===")
        print(stats)
        return stats

    def data_structure(self):
        """Show data types, non-null counts, and missing values for all columns"""
        info_df = pd.DataFrame({
            'dtype': self.df.dtypes,
            'non_null_count': self.df.notnull().sum(),
            'missing_count': self.df.isnull().sum(),
            'unique_values': self.df.nunique()
        })
        print("\n=== Data Structure Summary ===")
        print(info_df)
        return info_df

    def column_overview(self):
        """Return lists of numeric, categorical, and boolean columns"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        overview = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'boolean_columns': bool_cols
        }
        print("\n=== Column Overview ===")
        print(overview)
        return overview
