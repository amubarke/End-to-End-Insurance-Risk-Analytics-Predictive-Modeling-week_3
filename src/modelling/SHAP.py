import shap
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

class ModelExplainability:
    """
    Generate SHAP and LIME explanations for classification or regression models.
    """
    def __init__(self, trained_pipeline, model_name, feature_names=None):
        self.pipeline = trained_pipeline
        self.model_name = model_name
        self.feature_names = feature_names
        self.model = self.pipeline.named_steps["model"]


    # ----------------------------
    # SHAP explanations
    # ----------------------------
    def shap_summary(self, X):
        """Global explanation: feature impact"""
        # Transform using pipeline preprocessing
        X_transformed = self.pipeline.named_steps["preprocessor"].transform(X)
        
        # Create SHAP explainer
        explainer = shap.Explainer(self.model, X_transformed)
        shap_values = explainer(X_transformed)

        # Plot summary
        shap.summary_plot(shap_values, X_transformed, feature_names=self.feature_names)
        return shap_values

    def shap_force_plot(self, X_row):
        """Local explanation: single-row prediction"""
        X_transformed = self.pipeline.named_steps["preprocessor"].transform(X_row)
        explainer = shap.Explainer(self.model, X_transformed)
        shap_values = explainer(X_transformed)

        # Force plot for the first row
        shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_transformed[0,:], matplotlib=True)
        plt.show()
        return shap_values

    # ----------------------------
    # LIME explanations
    # ----------------------------
    def lime_explanation(self, X_row, class_names=None):
        """
        Local human-readable explanation using LIME
        - X_row: single-row DataFrame
        - class_names: list of str, optional class names for classification
        """
        X_array = X_row.values
        # Extract training data for LIME background
        if hasattr(self.pipeline.named_steps["preprocessor"], "transform"):
            # Use preprocessor on the full training set if needed
            raise NotImplementedError("Provide training data for LimeTabularExplainer")

        # Default LimeTabularExplainer
        explainer = LimeTabularExplainer(
            training_data=X_array,
            feature_names=X_row.columns.tolist(),
            class_names=class_names,
            mode="classification"
        )
        exp = explainer.explain_instance(X_array[0], self.pipeline.predict_proba, num_features=10)
        exp.show_in_notebook(show_table=True)
        return exp
