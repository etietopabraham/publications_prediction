from predicting_publications.constants import *
from predicting_publications.utils.common import read_yaml, create_directories
from predicting_publications import logger
from predicting_publications.entity.config_entity import (DataIngestionConfig, 
                                                          DataValidationConfig,
                                                          DataTransformationConfig)

import os

class ConfigurationManager:
    """
    ConfigurationManager manages configurations needed for the data pipeline.

    The class reads configuration, parameter, and schema settings from specified files
    and provides a set of methods to access these settings. It also takes care of
    creating necessary directories defined in the configurations.

    Attributes:
    - config (dict): Configuration settings.
    - params (dict): Parameters for the pipeline.
    - schema (dict): Schema information.
    """
    
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH, 
                 params_filepath = PARAMS_FILE_PATH, 
                 schema_filepath = SCHEMA_FILE_PATH,
                 feature_schema_filepath = FEATURE_SCHEMA_FILE_PATH) -> None:
        """
        Initialize ConfigurationManager with configurations, parameters, and schema.

        Args:
        - config_filepath (Path): Path to the configuration file.
        - params_filepath (Path): Path to the parameters file.
        - schema_filepath (Path): Path to the schema file.

        Creates:
        - Directories specified in the configuration.
        """
        self.config = self._read_config_file(config_filepath, "config")
        self.params = self._read_config_file(params_filepath, "params")
        self.schema = self._read_config_file(schema_filepath, "initial_schema")
        self.feature_schema_filepath = self._read_config_file(feature_schema_filepath, "feature_engineered_schema")

        # Create the directory for storing artifacts if it doesn't exist
        create_directories([self.config.artifacts_root])

    def _read_config_file(self, filepath: str, config_name: str) -> dict:
        """
        Read a configuration file and return its content.

        Args:
        - filepath (str): Path to the configuration file.
        - config_name (str): Name of the configuration (for logging purposes).

        Returns:
        - dict: Configuration settings.

        Raises:
        - Exception: If there's an error reading the file.
        """
        try:
            return read_yaml(filepath)
        except Exception as e:
            logger.error(f"Error reading {config_name} file: {filepath}. Error: {e}")
            raise
    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Extract and return data ingestion configurations as a DataIngestionConfig object.

        This method fetches settings related to data ingestion, like directories and file paths,
        and returns them as a DataIngestionConfig object.

        Returns:
        - DataIngestionConfig: Object containing data ingestion configuration settings.

        Raises:
        - AttributeError: If the 'data_ingestion' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_ingestion
            # Create the root directory for data ingestion if it doesn't already exist
            create_directories([config.root_dir])
            
            return DataIngestionConfig(
                root_dir=Path(config.root_dir),
                local_data_file=Path(config.local_data_file),
            )

        except AttributeError as e:
            logger.error("The 'data_ingestion' attribute does not exist in the config file.")
            raise e
        

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Extract and return data validation configurations as a DataValidationConfig object.

        This method fetches settings related to data validation, like directories, file paths,
        and schema, and returns them as a DataValidationConfig object.

        Returns:
        - DataValidationConfig: Object containing data validation configuration settings.

        Raises:
        - AttributeError: If the 'data_validation' attribute does not exist in the config file.
        """
        try:
            # Extract data validation configurations
            config = self.config.data_validation
            
            # Extract schema for data validation
            schema = self.schema.columns
            
            # Ensure the parent directory for the status file exists
            create_directories([os.path.dirname(config.status_file)])

            
            # Construct and return the DataValidationConfig object
            return DataValidationConfig(
                root_dir=Path(config.root_dir),
                data_source_file=Path(config.data_source_file),
                status_file=Path(config.status_file),
                initial_schema=schema
            )

        except AttributeError as e:
            # Log the error and re-raise the exception for handling by the caller
            logger.error("The 'data_validation' attribute does not exist in the config file.")
            raise e


    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Extract and return data transformation configurations as a DataTransformationConfig object.

        This method fetches settings related to data transformation, like directories and file paths,
        and returns them as a DataTransformationConfig object.

        Returns:
        - DataTransformationConfig: Object containing data transformation configuration settings.

        Raises:
        - AttributeError: If the 'data_transformation' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_transformation
            
            # Ensure the root directory for data transformation exists
            create_directories([config.root_dir])

            # Construct and return the DataTransformationConfig object
            return DataTransformationConfig(
                root_dir=Path(config.root_dir),
                data_source_file=Path(config.data_source_file),
                data_validation=Path(config.data_validation),
            )

        except AttributeError as e:
            # Log the error and re-raise the exception for handling by the caller
            logger.error("The 'data_transformation' attribute does not exist in the config file.")
            raise e

