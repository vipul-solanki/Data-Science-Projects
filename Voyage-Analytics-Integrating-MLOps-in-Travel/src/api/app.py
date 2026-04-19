from flask import Flask, request, jsonify
import joblib
import pandas as pd

from src.models.recommender import build_recommender, recommend_hotels

print("🔥 Flask file loaded")

app = Flask(__name__)

# Load models
reg_model = joblib.load("flight_price_model.pkl")
clf_model = joblib.load("gender_model.pkl")

# Load recommender data
df, hotels_raw = build_recommender()


@app.route("/")
def home():
    return "Travel MLOps API Running 🚀"


@app.route("/predict-flight-price", methods=["POST"])
def predict_price():
    data = request.json
    input_df = pd.DataFrame([data])

    prediction = reg_model.predict(input_df)[0]

    return jsonify({"predicted_price": float(prediction)})


@app.route("/predict-gender", methods=["POST"])
def predict_gender():
    data = request.json
    input_df = pd.DataFrame([data])

    prediction = clf_model.predict(input_df)[0]

    return jsonify({"predicted_gender": int(prediction)})


@app.route("/recommend-hotels", methods=["POST"])
def recommend():
    data = request.json
    user_id = data.get("user_id")
    location = data.get("location")

    recs = recommend_hotels(user_id, location, df, hotels_raw)

    return jsonify(recs.to_dict(orient="records"))


if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)