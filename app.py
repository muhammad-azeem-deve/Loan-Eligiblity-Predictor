import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request, send_file
import os
from io import BytesIO

app = Flask(__name__)

def load_and_preprocess_data():
    file_path = 'loan_approval_dataset.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found.")

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    X = data.drop(['loan_id', 'loan_status'], axis=1)
    y = data['loan_status'].map(lambda x: 1 if str(x).lower() in ['approved', '1'] else 0)

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    num_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    feature_order = X.columns.tolist()

    return X, y, label_encoders, scaler, num_imputer, cat_imputer, categorical_cols, numerical_cols, feature_order

def train_model(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load and train
X, y, label_encoders, scaler, num_imputer, cat_imputer, categorical_cols, numerical_cols, feature_order = load_and_preprocess_data()
model = train_model(X, y)

@app.route('/')
def home():
    return render_template('index.html', categorical_cols=categorical_cols, numerical_cols=numerical_cols)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method != 'POST':
        return "Method Not Allowed", 405

    features = {}
    for col in categorical_cols:
        value = request.form.get(col)
        if not value:
            return render_template('error.html', message=f"Missing value for {col}")
        features[col] = value
    for col in numerical_cols:
        value = request.form.get(col)
        if not value or not value.replace('.', '', 1).isdigit():
            return render_template('error.html', message=f"Invalid or missing value for {col}")
        features[col] = float(value)

    input_data = pd.DataFrame([features])

    # Keep original for explanation
    raw_input = input_data.copy()

    # Preprocess
    input_data[numerical_cols] = num_imputer.transform(input_data[numerical_cols])
    input_data[categorical_cols] = cat_imputer.transform(input_data[categorical_cols])

    for col in categorical_cols:
        val = features[col]
        if val in label_encoders[col].classes_:
            input_data[col] = label_encoders[col].transform([val])[0]
        else:
            input_data[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]

    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    input_data = input_data[feature_order]

    prediction = model.predict(input_data)[0]
    result = 'Approved' if prediction == 1 else 'Not Approved'

    # ======== Generate reason ========
    reasons = []

    # Example thresholds (adjust as needed)
    income = raw_input.get('income', [0])[0]
    cibil = raw_input.get('cibil_score', [0])[0]
    dependents = raw_input.get('no_of_dependents', [0])[0]
    commercial = raw_input.get('commercial_asset_value', [0])[0]
    bank = raw_input.get('bank_asset_value', [0])[0]

    if result == "Approved":
        if cibil >= 750:
            reasons.append("Strong CIBIL score.")
        if income >= 30000:
            reasons.append("Stable monthly income.")
        if bank > 50000:
            reasons.append("Good bank asset value.")
        if dependents <= 2:
            reasons.append("Lower number of dependents.")
        if commercial >= 100000:
            reasons.append("Valuable commercial assets.")
    else:
        if cibil < 600:
            reasons.append("Low CIBIL score.")
        if income < 20000:
            reasons.append("Insufficient monthly income.")
        if bank < 20000:
            reasons.append("Low bank asset value.")
        if dependents > 3:
            reasons.append("High number of dependents.")
        if commercial < 50000:
            reasons.append("Insufficient commercial assets.")

    reason_text = "\n".join(reasons) if reasons else "Model did not find strong enough indicators."

    # ======== Final result file ========
    result_text = f"Loan Prediction Result: {result}\n\nReasons:\n{reason_text}\n\nProvided Details:\n"
    for key, value in features.items():
        result_text += f"{key}: {value}\n"

    buffer = BytesIO()
    buffer.write(result_text.encode())
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='loan_prediction_result.txt', mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
