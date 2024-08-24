from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from pycaret.anomaly import *

app = Flask(__name__)

# load the model
model = load_model('anomaly_detection')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get data from the request
    data1 = request.form["div_name"]
    data2 = request.form["merchant"]
    data3 = request.form["cat_desc"]
    data4 = request.form["trans_dt"]
    data5 = request.form["amt"]

    # convert amount to float
    try:
        data5 = float(data5)
    except ValueError:
        return "Invalid input for amount. Please enter a valid number.", 400
    
    # log the data for debugging
    print(f"Data received: {data1}, {data2}, {data3}, {data4}, {data5}")

    # convert the input data into a df with proper column names
    input_data = pd.DataFrame([[data1, data2, data3, data4, data5]],
                              columns=['DIV_NAME', 'MERCHANT', 'CAT_DESC', 'TRANS_DT', 'AMT'])

    # log df for debugging
    print("Input DataFrame:\n", input_data)

    # make predictions using the model
    prediction_df = predict_model(model,input_data)

    # log the prediction result for debugging
    print("Prediction result:\n", prediction_df)

    # assign variable to the Anomaly result
    prediction = prediction_df['Anomaly'].iloc[0]

    # check the prediction and return appropriate message
    if prediction == 0:
        prediction_text = "This record is NOT an anomaly."
    else:
        prediction_text = "This record is an anomaly."

    # return the prediction to the template
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)

