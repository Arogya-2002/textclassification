import os
import pandas as pd
import sys
import joblib
from src.exceptions.exceptions import CustomException
from src.Logger.logger import logging
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from sklearn.preprocessing import LabelEncoder

from dataclasses import dataclass
from indic_transliteration.sanscript import transliterate, TELUGU, HK



@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts',"data.csv")
    label_map_path: str = os.path.join('artifacts',"label_map.pkl")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def preprocess_tenglish(self, text):
         # Transliterate Telugu to English
        try:
           
            text = transliterate(text, TELUGU, HK)  # Telugu script to phonetic English
            

        except Exception as e:
            logging.error(f"Error in transliterating: {e}")
            return None
            
    # Lowercase and normalize
        text = text.lower()
        return text
    

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df=pd.read_csv('notebook\data\data.csv')

            logging.info('Read the dataset as dataframe')
            logging.info("converting telugu to english ")

            df['processed_text'] = df['text'].apply(self.preprocess_tenglish)
            df.drop(columns=['text'],inplace=True)

            logging.info("converted telugu to english ")

            label_encoder = LabelEncoder()
            label_encoder.fit(df['emotion_label'])

            labels = df['emotion_label'].unique()
            label_map = {label: idx for idx, label in enumerate(labels)}
            df['encoded_label'] = df['emotion_label'].map(label_map) 

            


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Saving the label map to artifacts path")
            joblib.dump(label_encoder,self.ingestion_config.label_map_path)


            return(
                self.ingestion_config.raw_data_path
               
            )
        except Exception as e:
            raise CustomException(e,sys)
        




        
        
if __name__=="__main__":
    obj=DataIngestion()
    data_path=obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    X_train,y_train,X_test,y_test = data_transformation_obj.data_transformation(data_path)

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.model_tuner(X_train, y_train, X_test, y_test))  
