import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    Handles:
    - Cleaning
    - Missing value handling
    - Encoding categorical features
    - Train-test split
    """

    def __init__(self, df):
        self.df = df.copy()
        self.categorical_cols = self.df.select_dtypes(include=["category", "object"]).columns
        self.numeric_cols = self.df.select_dtypes(include=["float64", "int64", "int32"]).columns

    def preprocess(self):
        # Basic missing value handling
        df = self.df.fillna({
            col: self.df[col].mode()[0] if col in self.categorical_cols else self.df[col].median()
            for col in self.df.columns
        })

        return df

    def build_preprocessing_pipeline(self):
        """Return a sklearn ColumnTransformer for consistent preprocessing."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols),
                ("num", "passthrough", self.numeric_cols)
            ]
        )
        return preprocessor

    def split(self, df, target, test_size=0.2):
        X = df.drop(target, axis=1)
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=42)
