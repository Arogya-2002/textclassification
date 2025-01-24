from flask import Flask, request, render_template, jsonify
from src.pipeline.predict_pipeline import PredictPipeline  # Adjust the import as per your project structure
import pandas as pd

app = Flask(__name__)

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get text input from the form
        text_data = request.form.get('text')
        
        if text_data:
            # Convert the text input into a DataFrame and make a prediction
            input_df = pd.DataFrame({'text': [text_data]})
            prediction = predict_pipeline.predict(input_df)
            # If prediction is a list or array, take the first element
            prediction = prediction[0] if isinstance(prediction, list) else prediction
    
    # Render the HTML template and pass the prediction result
    return render_template('index.html', prediction=prediction)

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080) 
