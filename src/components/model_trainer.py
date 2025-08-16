import os 
import sys 
from dataclasses import dataclass

from src.logger import logging 
from src.exception import CustomException 
from src.utlis import save_object

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Base models
            models = {
                "random forest": RandomForestRegressor(),
                "decision tree": DecisionTreeRegressor(),
                "gradient boosting": GradientBoostingRegressor(),
                "linear regression": LinearRegression(),
                "k neighbours regressor": KNeighborsRegressor(),
                "xg boost regressor": XGBRFRegressor(),
                "cat boost regressor": CatBoostRegressor(verbose=False),
                "adaboost regressor": AdaBoostRegressor(),
            }

            # Hyperparameter grids
            params = {
                "random forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },
                "decision tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10],
                },
                "gradient boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                },
                "linear regression": {},  # no params
                "k neighbours regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "xg boost regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                },
                "cat boost regressor": {
                    "depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                    "iterations": [200, 500],
                },
                "adaboost regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1, 1],
                },
            }

            model_report = {}
            best_model = None
            best_model_name = None
            best_model_score = -1

            for name, model in models.items():
                logging.info(f"Tuning hyperparameters for {name}...")
                param_grid = params[name]

                if param_grid:  # if params exist
                    gs = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0)
                    gs.fit(X_train, y_train)
                    tuned_model = gs.best_estimator_
                else:  # e.g. linear regression
                    tuned_model = model
                    tuned_model.fit(X_train, y_train)

                # Evaluate on test set
                y_pred = tuned_model.predict(X_test)
                score = r2_score(y_test, y_pred)
                model_report[name] = score

                logging.info(f"{name} R2 Score: {score}")

                # Track best
                if score > best_model_score:
                    best_model_score = score
                    best_model_name = name
                    best_model = tuned_model

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best model: {best_model_name} with R2 score {best_model_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
