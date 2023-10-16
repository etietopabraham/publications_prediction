from predicting_publications import logger
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os 

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from predicting_publications.config.configuration import ModelTrainerConfig

class ModelTrainer:
    """
    ModelTrainer class handles the training of the GradientBoostingRegressor model.

    This component reads in the transformed training and test data, trains a Gradient 
    Boosting Regressor model using the specified hyperparameters, and saves the trained 
    model to the specified path.

    Attributes:
    - config (ModelTrainerConfig): Configuration settings for the model training process.
    """

    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize ModelTrainer with the given configurations.

        Args:
        - config (ModelTrainerConfig): Configuration settings for model training.
        """
        self.config = config

    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
        - X_train: Training data features.
        - y_train: Training data target.

        Returns:
        - Best hyperparameters found during the search.
        """
        param_dist = {
            'n_estimators': sp_randint(50, 200),
            'max_depth': sp_randint(1, 10),
            'learning_rate': sp_uniform(0.01, 0.2),
            'subsample': sp_uniform(0.5, 0.5),
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': sp_randint(2, 20),
            'min_samples_leaf': sp_randint(1, 20)
        }

        n_iter_search = 10  # Reduced to 10 combinations
        random_search = RandomizedSearchCV(GradientBoostingRegressor(),
                                           param_distributions=param_dist,
                                           n_iter=n_iter_search,
                                           cv=3,  # Reduced to 3-fold cross-validation
                                           scoring='neg_mean_squared_error',
                                           verbose=2,
                                           n_jobs=-1)

        random_search.fit(X_train, y_train)

        return random_search.best_params_

    def train(self):
        """
        Train a Gradient Boosting Regressor model.

        This method:
        1. Loads the training and test data from the paths specified in the configuration.
        2. Separates the predictors and target variables.
        3. Initializes a Gradient Boosting Regressor model with the specified hyperparameters.
        4. Fits the model on the training data.
        5. Saves the trained model to the path specified in the configuration.
        """
        # Load training dataset
        train_data = pd.read_csv(self.config.train_data_path)

        # Separate predictors and target variable
        X_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]].values.ravel()

        # Perform hyperparameter tuning
        # best_params = self.hyperparameter_tuning(X_train, y_train)

        # Log the best parameters
        # logger.info(f"Best hyperparameters found: {best_params}")

        # Best hyperparameters
        best_params = {
            'learning_rate': self.config.learning_rate,
            'max_depth': self.config.max_depth,
            'max_features': None if self.config.max_features == 'None' else self.config.max_features,
            'min_samples_leaf': self.config.min_samples_leaf,
            'min_samples_split': self.config.min_samples_split,
            'n_estimators': self.config.n_estimators,
            'subsample': self.config.subsample,
            'random_state': self.config.random_state
        }

        # Train the model with the best parameters
        gb_model = GradientBoostingRegressor(**best_params)
        gb_model.fit(X_train, y_train)

        # Save the trained model
        model_save_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(gb_model, model_save_path)
        logger.info(f"Model saved successfully to {model_save_path}")
