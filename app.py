from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("random_forest_model.pkl")

# Load dataset to get feature names
data = pd.read_csv("heart.csv")
feature_names = list(data.drop(columns=['target']).columns)

# Mapping technical -> user-friendly labels
friendly_labels = {
    "age": "Age (years)",
    "sex": "Gender (0 = Female, 1 = Male)",
    "cp": "Chest Pain Type (0–3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "fbs":"Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)",
    "restecg": "Resting ECG Results (0–2)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1 = Yes, 0 = No)",
    "oldpeak": "ST Depression (exercise vs rest)",
    "slope": "Slope of Peak Exercise ST Segment (0–2)",
    "ca": "Major Vessels Colored by Fluoroscopy (0–3)",
    "thal": "Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)"
}

@app.route('/')
def home():
    return render_template('index.html', features=feature_names, labels=friendly_labels)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect values from form
        values = [float(request.form[feature]) for feature in feature_names]
        values = np.array(values).reshape(1, -1)

        # Prediction
        prediction = model.predict(values)[0]
        probability = model.predict_proba(values)[0][1] * 100

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        # --- Find top contributing factors ---
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'value': values[0],
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False)

        # Pick top 3 features
        top_reasons = importance_df.head(3).to_dict(orient='records')

        # Replace technical names with user-friendly ones
        for r in top_reasons:
            r['feature'] = friendly_labels[r['feature']]

        return render_template('result.html',
                               result=result,
                               prob=round(probability, 2),
                               reasons=top_reasons)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
