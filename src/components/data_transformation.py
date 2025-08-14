import os 
import sys 
from src.logger import logging 
from src.exception import CustomException
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from src.utlis import save_object

@dataclass

class DataTransformationConfig:
    preprocessor_ob_file_path:str=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''this function does the data transformation'''
        try:
            numerical_features=['writing_score','reading_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info("numerical coulum scaling completed")
            categorical_peipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('categorical coulumns encoding completed')

            preprocessor=ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_features),
                ('cat_pipeline',categorical_peipeline,categorical_columns)
            
            ])
            logging.info('joined categorical and numerical feature using column transformerr')

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        

    def intiate_data_transformation(self,train_path,test_path):

        try :
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('the train and test path data are sucessfully loaded ')

            preprocessor_obj=self.get_data_transformer_object()
            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('applying dataframe transformer on train and testing ')
            

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj


            )
            logging.info('saving preprocessed object')
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        


