from flask import Flask, request, url_for, redirect, render_template, jsonify

# from pycaret.regression import *
from pycaret.classification import *

# from pycaret.anomaly import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

amber_model = load_model("templates/amber/mushroom-pipeline")
amber_cols = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attached",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

# jav_model = load_model("deployment_28042020")
# jav_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

# gavin_model = load_model("deployment_28042020")
# gavin_cols = ["age", "sex", "bmi", "children", "smoker", "region"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/mushroom-poison-detector")
def load_nushroom():

    return render_template("amber/mushroom_poison_detector.html")


@app.route("/mushroom-poison-detector", methods=["POST"])
def mushroom_page():
    features = [x for x in request.form.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns=amber_cols)
    # print(data_unseen['bruises'])

    int_columns = ['bruises', 'gill-attached', 'ring-number']  # Replace with actual column names
    data_unseen[int_columns] = data_unseen[int_columns].astype(int)

    prediction = predict_model(amber_model, data=data_unseen)
    prediction = prediction["prediction_label"].values[0]

    return render_template(
        "amber/mushroom_poison_detector.html",
        pred=prediction
    )


@app.route("/predict-mushroom-api", methods=["POST"])
def predict_mushroom_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(amber_model, data=data_unseen)
    output = prediction["prediction_label"]
    return jsonify(output)


# @app.route("/house-price-prediction")
# def house_page():
#     int_features = [x for x in request.form.values()]
#     final = np.array(int_features)
#     data_unseen = pd.DataFrame([final], columns=jav_cols)
#     prediction = predict_model(jav_model, data=data_unseen, round=0)
#     prediction = int(prediction.Label[0])

#     return render_template(
#         "templates/javerine/house_price_prediction.html",
#         pred="Expected Bill will be {}".format(prediction),
#     )


# @app.route("/predict-house-api", methods=["POST"])
# def predict_house_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(jav_model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
