from predicting_publications import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from predicting_publications.config.configuration import DataTransformationConfig

class DataTransformation:
    """
    Handles the transformation of the ingested dataset, generating temporal features, 
    aggregating the data, and splitting it into training and validation sets.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation component by reading the data source file 
        specified in the config.

        Args:
        - config (DataTransformationConfig): Configuration settings for data transformation.

        Attributes:
        - df (pd.DataFrame): The data to be transformed.
        """
        self.config = config
        try:
            self.df = pd.read_csv(self.config.data_source_file)
        except FileNotFoundError:
            logger.error(f"File not found: {self.config.data_source_file}")
            raise

    def generate_temporal_features_and_aggregate(self):
        # Convert the 'timestamp' column to a datetime format if it's not already
        if self.df['timestamp'].dtype != 'datetime64[ns]':
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='s')

        """
        Generate temporal features and aggregate the dataset.
        """
        # Generating temporal features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day'] = self.df['timestamp'].dt.day
        self.df['dayofweek'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month

        # Aggregating data by hour and location
        agg_columns = {
            'likescount': 'mean',
            'commentscount': 'mean',
            'symbols_cnt': 'mean',
            'words_cnt': 'mean',
            'hashtags_cnt': 'mean',
            'mentions_cnt': 'mean',
            'links_cnt': 'mean',
            'emoji_cnt': 'mean'
        }

        logger.info("Grouping data by timestamp, lon, lat, hour, day, day of week, and month")
        self.grouped_data = self.df.groupby(['timestamp', 'lon', 'lat', 'hour', 'day', 'dayofweek', 'month']).agg(agg_columns).reset_index()
        
        logger.info("Setting publication count grouped by timestamp, lon, and lat")
        self.grouped_data['publication_count'] = self.df.groupby(['timestamp', 'lon', 'lat']).size().values

    def split_data_into_train_and_test(self):
        """
        Split the aggregated data into training and test sets.
        """
        # Drop 'timestamp' as it's strongly correlated with other time features and may cause data leakage
        X = self.grouped_data.drop(['publication_count', 'timestamp'], axis=1)
        y = self.grouped_data['publication_count']

        # Split the data into training and validation sets and set them as class attributes
        logger.info("Splitting data into train and test values")
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training data shape: {self.X_train.shape}, Validation data shape: {self.X_val.shape}")
        print(f"Training data shape: {self.X_train.shape}, Validation data shape: {self.X_val.shape}")

    def _save_datasets(self, train_filename: str, test_filename: str):
        """
        Save the train and test datasets to the output path specified in the configuration.

        Args:
        - train_filename (str): Name of the file to save the training data.
        - test_filename (str): Name of the file to save the test data.
        """
        train_output_path = self.config.root_dir / train_filename
        test_output_path = self.config.root_dir / test_filename
        
        try:
            # Save training data
            train_data = pd.concat([self.X_train, self.y_train], axis=1)
            train_data.to_csv(train_output_path, index=False)
            logger.info(f"Training Data saved successfully to {train_output_path}")

            # Save test data
            test_data = pd.concat([self.X_val, self.y_val], axis=1)
            test_data.to_csv(test_output_path, index=False)
            logger.info(f"Test Data saved successfully to {test_output_path}")

        except Exception as e:
            logger.error(f"Error while saving the datasets: {e}")
            raise

    def orchestrate_transformation(self, train_filename: str = "train_data.csv", test_filename: str = "test_data.csv"):
        """
        Orchestrates the data transformation process by:
        1. Generating temporal features and aggregating the data.
        2. Splitting data into training and test sets.
        3. Saving the training and test datasets.

        Args:
        - train_filename (str): Name of the file to save the training data. Default is "train_data.csv".
        - test_filename (str): Name of the file to save the test data. Default is "test_data.csv".
        """
        self.generate_temporal_features_and_aggregate()
        self.split_data_into_train_and_test()
        self._save_datasets(train_filename, test_filename)