import pandas as pd
from predicting_publications import logger
from predicting_publications.entity.config_entity import DataValidationConfig


class DataValidation:
    """
    Validates the data against a predefined schema to ensure that all expected columns 
    are present and of the correct type.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Initializes the DataValidation component by reading the data source file 
        specified in the config.

        Args:
        - config (DataValidationConfig): Configuration settings for data validation.

        Attributes:
        - df (pd.DataFrame): The data to be validated.
        """
        self.config = config
        try:
            self.df = pd.read_csv(self.config.data_source_file)
        except FileNotFoundError:
            logger.error(f"File not found: {self.config.data_source_file}")
            raise

    def validate_all_features(self) -> bool:
        """
        Validates that all expected columns are present in the dataframe.

        Returns:
        - bool: True if validation is successful, False otherwise.
        """
        validation_status = True
        status_message = "Validation status: "

        # Determine missing or extra columns
        all_columns = set(self.df.columns)
        expected_columns = set(self.config.initial_schema.keys())

        missing_columns = expected_columns - all_columns
        extra_columns = all_columns - expected_columns

        # Log and update the status message for any discrepancies
        if missing_columns:
            validation_status = False
            logger.warning(f"Missing columns: {', '.join(missing_columns)}")
            status_message += f"Missing columns: {', '.join(missing_columns)}. "
        if extra_columns:
            validation_status = False
            logger.warning(f"Extra columns found: {', '.join(extra_columns)}")
            status_message += f"Extra columns found: {', '.join(extra_columns)}. "
        if validation_status:
            logger.info("All expected columns are present in the dataframe.")
            status_message += "All expected columns are present."

        # Append the validation status to the file
        self._write_status_to_file(status_message, overwrite=True)
        return validation_status

    def validate_data_types(self) -> bool:
        """
        Validates the data types of each column in the dataframe against 
        the expected data types specified in the schema.

        Returns:
        - bool: True if all data types match, False otherwise.
        """
        validation_status = True
        status_message = "Data type validation status: "

        expected_data_types = {col: self.config.initial_schema[col]['type'] for col in self.config.initial_schema}

        for column, dtype in expected_data_types.items():
            # Check if the column exists in the dataframe
            if column in self.df.columns:
                if not pd.api.types.is_dtype_equal(self.df[column].dtype, dtype):
                    validation_status = False
                    logger.warning(f"Data type mismatch for column '{column}': Expected {dtype} but got {self.df[column].dtype}")
                    status_message += f"Data type mismatch for column '{column}': Expected {dtype} but got {self.df[column].dtype}. "
            else:
                validation_status = False
                logger.warning(f"Column '{column}' not found in dataframe.")
                status_message += f"Column '{column}' not found in dataframe. "

        if validation_status:
            logger.info("All data types are as expected.")
            status_message += "All data types are as expected."

        # Append the validation status to the file
        self._write_status_to_file(status_message)
        return validation_status

    def _write_status_to_file(self, message: str, overwrite: bool = False):
        """
        Writes a given message to the status file specified in the config.

        Args:
        - message (str): The message to write.
        - overwrite (bool): If True, overwrites the file. If False, appends to the file.
        """
        mode = 'w' if overwrite else 'a'
        try:
            with open(self.config.status_file, mode) as f:
                f.write(message + "\n")
        except Exception as e:
            logger.error(f"Error writing to status file: {e}")
            raise

    def run_all_validations(self):
        """
        Executes all validations and writes the overall validation status.
        """
        feature_validation_status = self.validate_all_features()
        data_type_validation_status = self.validate_data_types()

        overall_status = "Overall Validation Status: "
        if feature_validation_status and data_type_validation_status:
            overall_status += "All validations passed."
        else:
            overall_status += "Some validations failed. Check the log for details."
        
        self._write_status_to_file(overall_status)

    def _save_dataframe(self):
        """
        Save the dataframe to the output path specified in the configuration.
        """
        try:
            self.df.to_csv(self.config.validated_data_file, index=False)
            logger.info(f"Data saved successfully to {self.config.root_dir}")
        except Exception as e:
            logger.error(f"Error while saving the dataframe: {e}")
            raise
