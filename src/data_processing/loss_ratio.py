import matplotlib.pyplot as plt
import seaborn as sns

class LossRatioVisualizer:
    def __init__(self, df):
        self.df = df.copy()
        self.df["LossRatio"] = self.df["TotalClaims"] / self.df["TotalPremium"]

    def _group_loss(self, column):
        return (
            self.df.groupby(column)
            .apply(lambda x: x["LossRatio"].mean())
            .reset_index(name="LossRatio")
            .sort_values("LossRatio")
        )

    def plot_loss_by_province(self):
        data = self._group_loss("Province")
        plt.figure(figsize=(12, 6))
        sns.barplot(data=data, x="Province", y="LossRatio")
        plt.title("Loss Ratio by Province")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_loss_by_vehicle_type(self):
        data = self._group_loss("VehicleType")
        plt.figure(figsize=(12, 6))
        sns.barplot(data=data, x="VehicleType", y="LossRatio")
        plt.title("Loss Ratio by Vehicle Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_loss_by_gender(self):
        data = self._group_loss("Gender")
        plt.figure(figsize=(7, 5))
        sns.barplot(data=data, x="Gender", y="LossRatio")
        plt.title("Loss Ratio by Gender")
        plt.tight_layout()
        plt.show()
