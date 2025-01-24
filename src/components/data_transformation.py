import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from dataclasses import dataclass
import joblib

from src.exceptions.exceptions import CustomException
from src.Logger.logger import logging

@dataclass
class DataTransformationConfig:
    tokenizer_path: str = os.path.join('artifacts',"tokenizer.pkl")
    max_sequence_length: int = 100

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        

    def data_transformation(self,data_path):
        try:
            logging.info("Initiating data transformation")
            logging.info(f"Reading the dataset from {data_path}")
            df = pd.read_csv(data_path)
            
            if 'processed_text' not in df.columns:
                raise CustomException("'processed_text' column not found in the dataset", sys)
            
            tokenizer = Tokenizer()
            # text_data = df['processed_text'].values
            tokenizer.fit_on_texts(df['processed_text'])

            logging.info("Saving tokenizer to artifcats path ")
            joblib.dump(tokenizer,self.transformation_config.tokenizer_path)


            X_encoded = tokenizer.texts_to_sequences(df['processed_text'])
            X_padded = pad_sequences(X_encoded, maxlen=self.transformation_config.max_sequence_length)

            if 'encoded_label' not in df.columns:
                raise CustomException("'encoded_label' column not found in the dataset", sys)
            y = df['encoded_label']
            X  = X_padded
       

            logging.info("Splitting the data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Data transformation complete")
            return(
                X_train,
                y_train,
                X_test,
                y_test
                    )
        
        except Exception as e:
            raise CustomException(e,sys)