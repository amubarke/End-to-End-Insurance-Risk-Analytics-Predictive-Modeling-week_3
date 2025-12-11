from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

class SeverityModel:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=42)
        }
        self.pipelines = {}

    def train(self, X_train, y_train):
        for name, model in self.models.items():
            pipe = Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            self.pipelines[name] = pipe

    def evaluate(self, X_test, y_test):
        results = {}
        for name, pipe in self.pipelines.items():
            preds = pipe.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            results[name] = rmse
        return results

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.train(X_train, y_train)
        return self.evaluate(X_test, y_test)
