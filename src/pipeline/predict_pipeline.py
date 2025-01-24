import sys
import pandas as pd
from src.exceptions.exceptions import CustomException

import os
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Load the trained model saved as .pkl
            model_path = os.path.join("artifacts", "model.pkl")  # Update the model path if needed
            model = joblib.load(model_path)  # Loading the model using joblib

            # Load the saved tokenizer (preprocessor) if it's saved separately
            tokenizer_path = os.path.join('artifacts', 'tokenizer.pkl')  # Path to saved tokenizer
            tokenizer = joblib.load(tokenizer_path)  # Load the tokenizer used during training

            # Load the label encoder (assuming you used a LabelEncoder during training)
            label_encoder_path = os.path.join('artifacts', 'label_map.pkl')
            label_encoder = joblib.load(label_encoder_path)



            # Preprocessing - tokenization and padding of the input text
            X_encoded = tokenizer.texts_to_sequences(features['text'])
            max_length = max([len(seq) for seq in X_encoded])  # You can set a fixed length if needed
            X_padded = pad_sequences(X_encoded, maxlen=max_length)

            # Predict the class using the model
            preds = model.predict(X_padded)
            
            # For multi-class classification, get the class with the highest probability
            predicted_class_index = preds.argmax(axis=-1)  # This assumes your model is multi-class

            predicted_class = label_encoder.inverse_transform(predicted_class_index)


            return predicted_class
        
        except Exception as e:
            raise CustomException(e, sys)
