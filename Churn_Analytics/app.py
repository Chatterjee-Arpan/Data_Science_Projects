# coding: utf-8
from pathlib import Path

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# --- Load artifacts once (faster than loading on every POST) ---
BASE_DIR = Path(__file__).resolve().parent
df_1 = pd.read_csv(BASE_DIR / "dataset_head.csv")
model = pickle.load(open(BASE_DIR / "model.sav", "rb"))

DEFAULTS = {
    "query1": "0",   # SeniorCitizen
    "query2": "79.65",  # MonthlyCharges
    "query3": "1023.55", # TotalCharges
    "query4": "Female",  # gender
    "query5": "Yes",     # Partner
    "query6": "No",      # Dependents
    "query7": "Yes",     # PhoneService
    "query8": "No",      # MultipleLines
    "query9": "Fiber optic",  # InternetService
    "query10": "No",     # OnlineSecurity
    "query11": "Yes",    # OnlineBackup
    "query12": "No",     # DeviceProtection
    "query13": "No",     # TechSupport
    "query14": "Yes",    # StreamingTV
    "query15": "Yes",    # StreamingMovies
    "query16": "Month-to-month",  # Contract
    "query17": "Yes",    # PaperlessBilling
    "query18": "Electronic check", # PaymentMethod
    "query19": "12",     # tenure
}

def _base_context():
    ctx = {k: v for k, v in DEFAULTS.items()}
    ctx.update({"output1": "", "output2": "", "churn_prob": None, "p": None, "high": False, "error": ""})
    return ctx

@app.route("/", methods=["GET", "POST"])
def index():
    ctx = _base_context()

    if request.method == "POST":
        # Preserve user's inputs (so they stay on the screen after submit)
        for i in range(1, 20):
            key = f"query{i}"
            ctx[key] = request.form.get(key, "").strip()

        # Validate & coerce numeric fields (helps avoid pd.cut/.astype errors)
        try:
            senior = int(ctx["query1"])
            monthly = float(ctx["query2"])
            total = float(ctx["query3"])
            tenure = int(ctx["query19"])
        except ValueError:
            ctx["error"] = "Please enter valid numbers for Senior Citizen (0/1), Monthly Charges, Total Charges, and Tenure."
            return render_template("home.html", **ctx)

        # Keep same feature pipeline as your original app
        data = [[
            senior,
            monthly,
            total,
            ctx["query4"], ctx["query5"], ctx["query6"], ctx["query7"],
            ctx["query8"], ctx["query9"], ctx["query10"], ctx["query11"], ctx["query12"],
            ctx["query13"], ctx["query14"], ctx["query15"], ctx["query16"], ctx["query17"],
            ctx["query18"], tenure
        ]]

        new_df = pd.DataFrame(
            data,
            columns=[
                "SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender",
                "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
                "PaymentMethod", "tenure"
            ],
        )

        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        # Ensure numeric charge columns are present & numeric (model was fit with them)
        df_2['MonthlyCharges'] = pd.to_numeric(df_2['MonthlyCharges'], errors='coerce').fillna(0.0)
        df_2['TotalCharges'] = pd.to_numeric(df_2['TotalCharges'], errors='coerce').fillna(0.0)


        # Group the tenure in bins of 12 months (same as original)
        labels = [f"{i} - {i + 11}" for i in range(1, 72, 12)]
        df_2["tenure_group"] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

        # Drop the original tenure column (same as original)
        df_2.drop(columns=["tenure"], axis=1, inplace=True)

        # One-hot encode features (same as original; model expects this shape)
        new_df__dummies = pd.get_dummies(
            df_2[
                [
                    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group"
                ]
            ]
        )
        # Predict on the last row (user input)
        # IMPORTANT: the trained model expects MonthlyCharges and TotalCharges to be present.
        X_full = pd.concat(
            [
                df_2[["MonthlyCharges", "TotalCharges"]].astype(float),
                new_df__dummies,
            ],
            axis=1,
        )

        X = X_full.tail(1)

        # Align to the exact feature set used at fit time (prevents sklearn feature-name mismatch errors)
        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=list(model.feature_names_in_), fill_value=0)

        pred = int(model.predict(X)[0])
        churn_prob = float(model.predict_proba(X)[:, 1][0])

        ctx["churn_prob"] = churn_prob
        ctx["p"] = churn_prob * 100.0
        ctx["high"] = (ctx["p"] >= 50.0)
        if pred == 1:
            ctx["output1"] = "This customer is likely to churn."
        else:
            ctx["output1"] = "This customer is likely to continue."

        ctx["output2"] = "Tip: month-to-month contracts, electronic check payments, and high monthly charges often correlate with churn."

    return render_template("home.html", **ctx)

if __name__ == "__main__":
    # Note: debug=True is handy during development. Turn it off for production.
    app.run(debug=True)