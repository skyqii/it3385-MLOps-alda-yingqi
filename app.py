from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from pycaret.anomaly import load_model

app = Flask(__name__)

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

    try:
        data5 = float(data5)
    except ValueError:
        return "Invalid input for amount. Please enter a valid number.", 400
    
    # Log the data for debugging
    print(f"Data received: {data1}, {data2}, {data3}, {data4}, {data5}")

    # Convert the input data into a DataFrame with proper column names
    # input_data = pd.DataFrame([[data1, data2, data3, data4, data5]],
    #                           columns=['DIV_NAME', 'MERCHANT', 'CAT_DESC', 'TRANS_DT', 'AMT'])
    
    data = np.array([data1, data2, data3, data4, data5])

    input_data = pd.DataFrame(data)

    # Log the DataFrame for debugging
    print("Input DataFrame:\n", input_data)

    # Make predictions using the model
    prediction_df = model.predict(input_data)

    # Log the prediction result for debugging
    print("Prediction result:\n", prediction_df)

    prediction = prediction_df['Anomaly'].iloc[0]

    # Check the prediction and return appropriate message
    if prediction == 0:
        prediction_text = "This record is NOT an anomaly."
    else:
        prediction_text = "This record is an anomaly."

    # Return the prediction to the template
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)

