from flask import Flask, request, url_for, redirect, render_template, jsonify

# from pycaret.regression import *
# from pycaret.classification import *

import os
import pycaret.classification as pc
import pycaret.regression as pr

# from pycaret.anomaly import *
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig

app = Flask(__name__, template_folder="../templates")

@hydra.main(config_path="../config", config_name="config.yaml")
def load_models(cfg: DictConfig):
    global variables 
    variables = cfg
    return variables

load_models()
# print(variables.model.mushroom.path)
# print(variables.model.housing.path)

house_model = pr.load_model(variables.model.housing.path)
mushroom_model = pc.load_model(variables.model.mushroom.path)

# gavin_model = load_model("deployment_28042020")
# gavin_cols = ["age", "sex", "bmi", "children", "smoker", "region"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/mushroom-poison-detector", methods=["POST", "get"])
def mushroom_page():
    if request.method == "POST":
        features = [x for x in request.form.values()]
        final = np.array(features)
        data_unseen = pd.DataFrame([final], columns=variables.columns.mushroom)

        int_columns = ["bruises"]
        data_unseen[int_columns] = data_unseen[int_columns].astype(int)

        prediction = pc.predict_model(mushroom_model, data=data_unseen)
        prediction = prediction["prediction_label"].values[0]

        return render_template("mushroom_poison_detector.html", pred=prediction)

    return render_template("mushroom_poison_detector.html")


@app.route("/predict-mushroom-api", methods=["POST"])
def predict_mushroom_api():
    features = [x for x in request.form.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns=variables.columns.mushroom)

    int_columns = ["bruises"]
    data_unseen[int_columns] = data_unseen[int_columns].astype(int)

    prediction = pc.predict_model(mushroom_model, data=data_unseen)
    prediction = prediction["prediction_label"].values[0]
    return jsonify(prediction)


@app.route("/house-price-prediction", methods=["POST", "get"])
def house_page():
    if request.method == "POST":
        int_features = [x for x in request.form.values()]
        final = np.array(int_features)
        data_unseen = pd.DataFrame([final], columns=variables.columns.housing)
        prediction = pr.predict_model(house_model, data=data_unseen, round=0)
        prediction = int(prediction.Label[0])

        return render_template("house_price_prediction.html",pred=prediction)
    return  render_template("house_price_prediction.html")
    

@app.route("/predict-house-api", methods=["POST"])
def predict_house_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = pr.predict_model(house_model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(debug=True, host="0.0.0.0", port=port)
