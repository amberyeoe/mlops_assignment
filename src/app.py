from flask import Flask, request, url_for, redirect, render_template, jsonify

# from pycaret.regression import *
# from pycaret.classification import *

import pycaret.classification as pc

# from pycaret.anomaly import *
import pandas as pd
import pickle
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
print(variables.model.mushroom.path)

mushroom_model = pc.load_model(variables.model.mushroom.path)

# jav_model = load_model("deployment_28042020")
# jav_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

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
        # print(data_unseen['bruises'])

        int_columns = ["bruises", "gill-attached", "ring-number"]
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
    # print(data_unseen['bruises'])

    int_columns = ["bruises", "gill-attached", "ring-number"]
    data_unseen[int_columns] = data_unseen[int_columns].astype(int)

    prediction = pc.predict_model(mushroom_model, data=data_unseen)
    prediction = prediction["prediction_label"].values[0]
    return jsonify(prediction)


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
    app.run(debug=False)
