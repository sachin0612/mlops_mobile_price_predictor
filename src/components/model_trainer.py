import os
import sys
from dataclasses import dataclass
from clearml import Task
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            ) 
            task=Task.init(project_name='Mobile_Price_Predictor',task_name='Experiment Tracking')
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "Adaboost Regressor": AdaBoostRegressor()
            }

            params={
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                    #'weights':['uniform','distance']
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Adaboost Regressor":{
                    'learning_rate':[.1,.01,.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }
            task.connect(params)
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                              models=models,param=params)
            
            task.connect(model_report)
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            task.upload_artifact(name='data file',artifact_object='artifacts/data.csv')
            task.upload_artifact(name='trained_model',artifact_object='artifacts/model.pkl')
            task.upload_artifact(name='preprocessed_object',artifact_object='artifacts/preprocessor.pkl')
            predicted=best_model.predict(x_test)

            r2_scores=r2_score(y_test,predicted)
            return r2_scores
        except Exception as e:
            raise CustomException(e,sys)