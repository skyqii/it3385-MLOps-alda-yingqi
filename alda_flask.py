############################################
############################################
# Name: Alda Almira
# Admin No.: 224526L
# Module Group: BA2201
############################################
############################################

from flask import Flask, request, render_template
import pickle
import catboost

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('mushroom-pipeline.pkl', 'rb'))

# Define the route for the main page
@app.route('/')
def home():
    return render_template('alda_input.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        cap_shape = request.form['cap-shape']
        cap_surface = request.form['cap-surface']
        cap_color = request.form['cap-color']
        bruises = request.form['bruises']
        odor = request.form['odor']
        gill_size = request.form['gill-size']
        population = request.form['population']
        
        # Example: create a feature vector for prediction
        features = [cap_shape, cap_surface, cap_color, bruises, odor, gill_size, population]
        
        # Example: Placeholder logic for prediction
        # prediction = 'edible'

        # Make prediction using the loaded model
        prediction = model.predict([features])

        # Convert numerical prediction to 'edible' or 'poisonous'
        output = 'Edible' if prediction[0] == 'e' else 'Poisonous'

        return render_template('alda_input.html', prediction_text=f'The mushroom is likely {output}.')
    
    except Exception as e:
        # Print the error to the console and return a message
        print(f"Error occurred: {e}")
        return "An error occurred during prediction. Please try again."

if __name__ == "__main__":
    app.run(debug=True)
