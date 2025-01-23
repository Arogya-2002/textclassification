import os
import pandas as pd
import sys
from src.exceptions.exceptions import CustomException
from src.Logger.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from indic_transliteration.sanscript import transliterate, TELUGU, HK


from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

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

            logging.info("converted telugu to english ")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        




        
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)