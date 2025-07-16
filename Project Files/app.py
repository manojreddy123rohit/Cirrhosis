from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the saved models
try:
    with open('normalizer.pkl', 'rb') as f:
        normalizer = pickle.load(f)
    logger.info("Normalizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading normalizer.pkl: {e}")
    normalizer = None

try:
    with open('random_forest_model.pkl', 'rb') as f:
        best_rf = joblib.load(f)
    logger.info("Random Forest model loaded successfully")
except Exception as e:
    logger.error(f"Error loading random_forest_model.pkl: {e}")
    best_rf = None

# Ordered list of input features used for training (without ID, target is Stage)
input_columns = [
    'N_Days', 'Status', 'Drug', 'Age', 'Sex',
    'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
    'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
    'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin'
]

# Encoding maps (exact mappings used during training)
encode_maps = {
    'Status': {'C': 0, 'CL': 1, 'D': 2},
    'Drug': {'D-penicillamine': 0, 'Placebo': 1, 'Other': 2},
    'Sex': {'M': 0, 'F': 1},
    'Ascites': {'N': 0, 'Y': 1},
    'Hepatomegaly': {'N': 0, 'Y': 1},
    'Spiders': {'N': 0, 'Y': 1},
    'Edema': {'N': 0, 'Y': 1, 'S': 2}
}

@app.route('/')
def main():
    return render_template('home.html', input_columns=input_columns, encode_maps=encode_maps)

@app.route('/about')
def about():
    return render_template('about.html', input_columns=input_columns, encode_maps=encode_maps)

@app.route('/contact')
def contact():
    return render_template('contact.html', input_columns=input_columns, encode_maps=encode_maps)

# Route to display the form
@app.route('/form', methods=['GET'])
def index():
    return render_template('index.html', input_columns=input_columns, encode_maps=encode_maps)


# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for col in input_columns:
            val = request.form.get(col, '').strip()
            if col in encode_maps:
                input_data[col] = val
            else:
                try:
                    input_data[col] = float(val) if val else 0.0
                    if col == 'Age':
                        logger.debug(f"Age entered in days: {input_data[col]}")
                except ValueError:
                    input_data[col] = 0.0
                    logger.warning(f"Invalid input for {col}, defaulted to 0.0")

        logger.debug(f"Raw input: {input_data}")

        input_df = pd.DataFrame([input_data])

        # Encode categorical features
        for col in encode_maps:
            input_df[col] = input_df[col].map(encode_maps[col])
            if input_df[col].isnull().any():
                logger.warning(f"Unknown value in {col}, defaulted to 0")
                input_df[col] = input_df[col].fillna(0)

        # Normalize
        if normalizer is None:
            raise Exception("Normalizer is not loaded")
        normalized_df = pd.DataFrame(normalizer.transform(input_df[input_columns]), columns=input_columns)

        # Predict
        if best_rf is None:
            raise Exception("Random Forest model not loaded")
        prediction = best_rf.predict(normalized_df)[0]
        interpretation = "Cirrhosis" if prediction == 1 else "No Cirrhosis (Stage 0)"

        logger.debug(f"Prediction: {prediction}")

        return render_template("result.html", prediction=prediction, interpretation=interpretation)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("error.html", error_message=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)