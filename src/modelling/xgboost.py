from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

class XGBoostClaimModel:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

        # Define the XGBoost model
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )

        # Final pipeline
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", self.model)
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.pipeline.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        return {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "Confusion Matrix": cm,
            "Classification Report": classification_report(y_test, preds)
        }
