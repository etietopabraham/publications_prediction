from predicting_publications import logger
from predicting_publications.config.configuration import ConfigurationManager
from predicting_publications.components.data_transformation import DataTransformation


class DataTransformationPipeline:
    """
    This pipeline handles the data transformation steps.

    After the data validation stage, it's essential to transform the data before moving on 
    to model training. This class orchestrates the transformation by generating temporal 
    features, aggregating data, and splitting it into training and test sets.

    Attributes:
        STAGE_NAME (str): The name of this pipeline stage.
    """
    
    STAGE_NAME = "Data Transformation Pipeline"

    def __init__(self):
        """
        Initializes the pipeline with a configuration manager.
        """
        self.config_manager = ConfigurationManager()

    def run_data_transformation(self):
        """
        Run the data transformation steps.
        
        This method orchestrates the data transformation functions.
        """
        try:
            logger.info("Fetching data transformation configuration...")
            data_transformation_config = self.config_manager.get_data_transformation_config()

            logger.info("Initializing data transformation process...")
            data_transformation = DataTransformation(config=data_transformation_config)

            logger.info("Executing data transformation...")
            data_transformation.orchestrate_transformation()

            logger.info("Data Transformation Pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error encountered during the data transformation: {e}")
    
    def run_pipeline(self):
        """
        Run the entire Data Transformation Pipeline.
        
        This method orchestrates the process of the data transformation and
        provides logs for each stage of the pipeline.
        """
        try:
            with open(self.config_manager.get_data_transformation_config().data_validation, "r") as f:
                content = f.read()

            if "Overall Validation Status: All validations passed." in content:
                logger.info("Starting the Data Transformation Pipeline.")
                logger.info(f">>>>>> Stage: {DataTransformationPipeline.STAGE_NAME} started <<<<<<")
                self.run_data_transformation()
                logger.info(f">>>>>> Stage {DataTransformationPipeline.STAGE_NAME} completed <<<<<< \n\nx==========x")
            else:
                logger.error("Data Transformation Pipeline aborted due to validation errors.")
        except Exception as e:
            logger.error(f"Error encountered during the {DataTransformationPipeline.STAGE_NAME}: {e}")
            raise e

if __name__ == '__main__':
    pipeline = DataTransformationPipeline()
    pipeline.run_pipeline()
