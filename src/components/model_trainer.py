import os 
import pickle 
import sys 
from src.logger import logging 
from src.exception import CustomException 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from dataclasses import dataclass
from src.utlis import save_object,evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initate_model_trainer(self, train_array, test_array):
        try:
            logging.info('splitting training and test input data ')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

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

            # Evaluate models
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info("Best model found, training it...")

            # ✅ retrain best model on full training set
            best_model.fit(X_train, y_train)

            # Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)

            logging.info(f"Best predicted model: {best_model_name} with score {score}")
            return score

        except Exception as e:
            raise CustomException(e, sys)
