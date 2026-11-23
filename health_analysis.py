
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load the health study dataset."""
    data = pd.read_csv(file_path)
    return data


def get_basic_statistics(data):
    """Calculate basic statistics for health measurements."""
    columns = ["age", "height", "weight", "systolic_bp", "cholesterol"]
    statistics = data[columns].describe()
    return statistics


class HealthAnalyzer:
    """Analyze health study data."""

    def __init__(self, data):
        self.data = data

    def average_blood_pressure(self):
        """Calculate average systolic blood pressure."""
        return self.data["systolic_bp"].mean()

    def average_age(self):
        """Calculate average age."""
        return self.data["age"].mean()

    def average_blood_pressure_by_smoking(self):
        """Compare blood pressure between smokers and non-smokers."""
        return self.data.groupby("smoker")["systolic_bp"].mean()

    def plot_blood_pressure_by_age(self):
        """Show blood pressure in relation to age."""
        plt.scatter(self.data["age"], self.data["systolic_bp"])
        plt.xlabel("Age")
        plt.ylabel("Systolic Blood Pressure")
        plt.title("Blood Pressure by Age")
        plt.show()

    def multiple_regression(self):
        """Estimate blood pressure using age and weight."""
        x = self.data[["age", "weight"]].to_numpy()
        y = self.data["systolic_bp"].to_numpy()

        x = np.column_stack((np.ones(len(x)), x))

        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]

        return coefficients
    
    def disease_by_sex(self):
        """Count disease cases by sex."""
        return self.data.groupby("sex")["disease"].sum()