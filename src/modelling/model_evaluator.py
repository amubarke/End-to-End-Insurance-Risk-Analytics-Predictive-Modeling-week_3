from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    """Evaluate classification (claim probability) and regression (claim severity) models."""

    def print_claim_results(self, results_dict):
        """
        Print results for claim probability (classification) models.
        Expected metrics: accuracy, roc_auc
        """
        print("\n====== 📊 CLAIM PROBABILITY MODEL RESULTS ======\n")
        
        for model_name, metrics in results_dict.items():
            print(f"\n🔹 Model: {model_name}")
            
            # Ensure metrics is a dictionary
            if not isinstance(metrics, dict):
                print(f"⚠ Invalid metrics format for {model_name}: {metrics}")
                continue
            
            print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics else "Accuracy: N/A")
            print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in metrics else "ROC-AUC: N/A")

    def print_severity_results(self, results_dict):
        """
        Print results for claim severity (regression) models.
        Expected metrics: RMSE, MAE, R2
        """
        print("\n====== 📊 CLAIM SEVERITY MODEL RESULTS ======\n")
        
        for model_name, metrics in results_dict.items():
            print(f"\n🔹 Model: {model_name}")

            # Ensure metrics is a dictionary
            if not isinstance(metrics, dict):
                print(f"⚠ Invalid metrics format for {model_name}: {metrics}")
                continue
            
            print(f"RMSE: {metrics.get('rmse', 'N/A'):.2f}" if 'rmse' in metrics else "RMSE: N/A")
            print(f"MAE: {metrics.get('mae', 'N/A'):.2f}" if 'mae' in metrics else "MAE: N/A")
            print(f"R²: {metrics.get('r2', 'N/A'):.4f}" if 'r2' in metrics else "R²: N/A")
