import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle


from src.Logger.logger import logging
from src.utils.utils import save_object

from dataclasses import dataclass
import os
import pandas as pd
import sys


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def build_model(self,hp):
     # Define parameters
        embedding_dim = hp.Int('embedding_dim', min_value=50, max_value=300, step=50)  # Hyperparameter search for embedding dimension
        lstm_units = hp.Int('lstm_units', min_value=64, max_value=512, step=64)  # Hyperparameter search for LSTM units
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)  # Hyperparameter search for dropout rate
        
        vocab_size = 10000  # Adjust as needed
        max_length = 5
        num_classes = 10
        

        model = Sequential()
    
    # Embedding layer
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
        model.add(SpatialDropout1D(0.2))

    # LSTM layer with variable units
        model.add(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    
    # Dense layer with variable size
        model.add(Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer with softmax activation for multi-class classification
        model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with a learning rate search
        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
        return model

# Initialize the KerasTuner RandomSearch for hyperparameter tuning
    def model_tuner(self, X_train, y_train, X_test, y_test):
        try:
            tuner = kt.RandomSearch(
                self.build_model,# Function to build the model
                objective='val_accuracy',  # Optimizing validation accuracy
                max_trials=5,  # Number of trials to test different hyperparameters
                executions_per_trial=3,  # Number of times to train the model per trial
                directory='kt_dir',  # Directory to store results
                project_name='hyperparameter_tuning'
            )
    
            # Perform the hyperparameter search on training data with validation data
            tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

            # Get the best hyperparameters and best model
            best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
            print(f"Best hyperparameters: {best_hyperparameters.values}")

            # Build the model with the best hyperparameters
            best_model = tuner.hypermodel.build(best_hyperparameters)

            # Train the best model
            history = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)
            # Save the model to a pickle file
            save_object(
               file_path = self.trainer_config.trained_model_file_path,
                obj = best_model
           )


            test_loss, test_acc = best_model.evaluate(X_test, y_test)
            return (
                test_acc,
                test_loss

            )



        except Exception as e:
            raise Exception(e,sys)
