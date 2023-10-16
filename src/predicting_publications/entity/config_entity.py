from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion process.
    
    Attributes:
    - root_dir: Directory where data ingestion artifacts are stored.
    - local_data_file: Path to the local file where the data is already saved.
    """
    root_dir: Path  # Directory where data ingestion artifacts are stored
    local_data_file: Path  # Path to the local file where the data is already saved


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for data validation process.
    
    This configuration class captures the necessary paths and directories 
    required for the validation of data both pre and post feature engineering.
    
    Attributes:
    - root_dir: Directory where data validation results and artifacts are stored.
    - data_source_file: Path to the file where the ingested or feature-engineered data is stored.
    - status_file: Path to the file that captures the validation status (e.g., success, errors encountered).
    - initial_schema: Dictionary holding all schema configurations. This can include initial data schema,
                  feature-engineered data schema, and any other relevant schema definitions.
    """
    
    root_dir: Path  # Directory for storing validation results and related artifacts
    data_source_file: Path  # Path to the ingested or feature-engineered data file
    status_file: Path  # File for logging the validation status
    initial_schema: Dict[str, Dict[str, str]]  # Dictionary containing initial schema configurations


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the data transformation process.
    
    This configuration class captures the necessary paths and directories 
    required for the transformation of data post-ingestion and pre-model training.
    
    Attributes:
    - root_dir: Directory where data transformation results and artifacts are stored.
    - data_source_file: Path to the file where the ingested data is stored that needs to be transformed.
    """
    
    root_dir: Path  # Directory for storing transformation results and related artifacts
    data_source_file: Path  # Path to the ingested data file for transformation
    data_validation: Path # Path to the validated output file


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for the model training process.
    
    This configuration class captures the necessary paths, directories, 
    and hyperparameters required for training the model.
    
    Attributes:
    - root_dir: Directory for storing trained model and related artifacts.
    - train_data_path: Path to the training data.
    - test_data_path: Path to the testing/validation data.
    - model_name: Name (or path) to save the trained model.
    - target_column: The column name of the target variable.
    - n_estimators: Number of boosting stages.
    - max_depth: Maximum depth of the individual regression estimators.
    - learning_rate: Step size for updating weights.
    - subsample: Fraction of samples used for fitting individual base learners.
    - random_state: Seed for reproducibility.
    - max_features: The number of features to consider for best split.
    - min_samples_split: Minimum number of samples required to split an internal node.
    - min_samples_leaf: Minimum number of samples required at a leaf node.
    """
    
    root_dir: Path  # Directory for storing model training results and related artifacts
    train_data_path: Path  # Path to train data
    test_data_path: Path  # Path to test data
    model_name: str  # Name or path where the trained model should be saved
    target_column: str  # The target column in the dataset
    n_estimators: int  # Number of boosting stages
    max_depth: int  # Maximum depth of the regression estimators
    learning_rate: float  # Learning rate
    random_state: int  # Seed for reproducibility
    subsample: float  # Fraction of samples for fitting individual base learners
    max_features: str  # Number of features to consider for best split
    min_samples_split: int  # Min samples required to split an internal node
    min_samples_leaf: int  # Min samples required at a leaf node
