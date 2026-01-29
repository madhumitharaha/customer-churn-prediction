import os
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
# Load artifacts
model = pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(MODEL_DIR, "columns.pkl"), "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":

        # 1️⃣ Collect raw user input
        input_data = {
            "tenure": float(request.form["tenure"]),
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"]),
            "gender": request.form["gender"],
            "SeniorCitizen": request.form["SeniorCitizen"],
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "Contract": request.form["Contract"],
            "InternetService": request.form["InternetService"],
            "PaymentMethod": request.form["PaymentMethod"],
        }

        # 2️⃣ Convert to DataFrame
        df = pd.DataFrame([input_data])

        # 3️⃣ One-hot encode categorical values
        df = pd.get_dummies(df)

        # 4️⃣ Align with training columns
        df = df.reindex(columns=columns, fill_value=0)

        # 5️⃣ Scale
        df_scaled = scaler.transform(df)

        # 6️⃣ Predict
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]

        prediction = "❌ Customer Likely to Churn" if pred == 1 else "✅ Customer Will Stay"
        probability = round(prob * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability)

if __name__ == "__main__":
    app.run(debug=True)