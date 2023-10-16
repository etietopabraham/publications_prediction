import os
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from predicting_publications.utils.common import save_json
from predicting_publications.config.configuration import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation class.

        Parameters:
        - config: Configuration parameters for model evaluation.
        """
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Calculate evaluation metrics for the model predictions.

        Parameters:
        - actual: Ground truth values.
        - pred: Predicted values by the model.

        Returns:
        - rmse: Root Mean Squared Error.
        - mae: Mean Absolute Error.
        - r2: R2 Score.
        - average_relative_error: Average Relative Error.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        epsilon = 1e-10
        relative_error = np.abs(pred - actual) / (pred + epsilon)
        average_relative_error = relative_error.mean()

        return rmse, mae, r2, average_relative_error

    def load_data(self):
        """
        Load test data and the trained model.
        """
        self.test_data = pd.read_csv(self.config.test_data_path)
        self.model = joblib.load(self.config.model_path)
        self.X_test = self.test_data.drop([self.config.target_column], axis=1)
        self.y_test = self.test_data[self.config.target_column]

    def log_into_mlflow(self):
        """
        Log model parameters, metrics, and the model itself into MLflow.
        """
        self.load_data()

        # Set the MLflow registry URI
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme

        # Start an MLflow tracking session
        with mlflow.start_run():
            predicted_qualities = self.model.predict(self.X_test)
            (rmse, mae, r2, average_relative_error) = self.eval_metrics(self.y_test, predicted_qualities)
            scores = {"rmse": rmse, "mae": mae, "r2": r2, "average_relative_error": average_relative_error}

            # Save evaluation metrics to a JSON file
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log parameters and metrics into MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("average_relative_error", average_relative_error)


            # Log the model into MLflow based on the type of tracking URL
            if tracking_url_type_score != "file":
                mlflow.sklearn.log_model(self.model, "model", registered_model_name="GradientBoostingRegressor")
            else:
                mlflow.sklearn.log_model(self.model, "model")
