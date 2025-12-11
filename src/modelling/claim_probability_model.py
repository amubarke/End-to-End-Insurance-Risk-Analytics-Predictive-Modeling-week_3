from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

class ClaimProbabilityModel:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(random_state=42)
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
            probs = pipe.predict_proba(X_test)[:, 1]
            results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, probs)
            }
        return results

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.train(X_train, y_train)
        return self.evaluate(X_test, y_test)
