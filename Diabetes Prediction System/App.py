from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('best_model_lr1.pkl')
scaler = joblib.load('scaler.pkl')  # Load the fitted scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data from the request
        data = request.form.to_dict()
        print(f"Received data: {data}")  # Debug print statement

        # Convert form data to DataFrame with correct column names
        df = pd.DataFrame([data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        print(f"DataFrame: {df}")

        # Scale the features using the loaded scaler
        X_scaled = scaler.transform(df)
        print(f"Scaled Data: {X_scaled}")

        # Make prediction
        prediction = model.predict(X_scaled)
        print(f"Prediction: {prediction}")

        # Map prediction to text
        prediction_text = 'Diabetic' if prediction[0] == 1 else 'Normal'

        # Render the template with prediction_text
        return render_template('index.html', prediction_text=prediction_text)
    
    except Exception as e:
        print(f"An error occurred: {e}")  # Print error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
