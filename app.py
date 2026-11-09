
from flask import Flask, request, jsonify, render_template, send_file
import pickle, pandas as pd, os, json, csv
from datetime import datetime

app = Flask(__name__)

with open("bank_subscription_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("metrics.json") as f:
    metrics = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        prediction = int(model.predict(df)[0])
        probability = model.predict_proba(df)[0]

        result = "YES" if prediction == 1 else "NO"

        exists = os.path.exists("history.csv")
        with open("history.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["Date", "Prediction", "Yes Probability", "No Probability"])
            writer.writerow([
                datetime.now(), result,
                round(probability[1]*100, 2),
                round(probability[0]*100, 2)
            ])

        return jsonify({
            "prediction": result,
            "yes_probability": round(probability[1]*100, 2),
            "no_probability": round(probability[0]*100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/metrics")
def get_metrics():
    return jsonify(metrics)

@app.route("/history")
def history():
    if not os.path.exists("history.csv"):
        return jsonify([])
    return jsonify(pd.read_csv("history.csv").fillna("").to_dict(orient="records"))

@app.route("/download-history")
def download_history():
    if not os.path.exists("history.csv"):
        return "No history available", 404
    return send_file("history.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
