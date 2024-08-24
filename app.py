from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from pycaret.anomaly import load_model

app = Flask(__name__)

# Load your pre-trained model

model = load_model('anomaly_detection')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data1 = request.form["div_name"]
    data2 = request.form["merchant"]
    data3 = request.form["cat_desc"]
    data4 = request.form["trans_dt"]
    data5 = request.form["amt"]
    
    # Convert the JSON data to a DataFrame
    data = np.array([data1,data2,data3,data4,data5])
    input_data = pd.DataFrame([data])
    
    # Get the prediction from the model
    prediction_df = model.predict(input_data)
    
    prediction = prediction_df['Anomaly'].iloc[0]

    # Ensure that prediction_df is a DataFrame
    if prediction == 0:
        prediction_text = "This record is NOT an anomaly."
    else:
        prediction_text = "This record is an anomaly."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
