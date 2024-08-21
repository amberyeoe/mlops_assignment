from flask import Flask, request, url_for, redirect, render_template, jsonify

import pandas as pd
import os
import pycaret.classification as pc
import pycaret.regression as pr
import pycaret.anomaly as pa

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

house_model = pr.load_model(variables.model.housing.path)
mushroom_model = pc.load_model(variables.model.mushroom.path)

# gavin_model = load_model("deployment_28042020")
# gavin_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

@app.route("/")
def home():
    return render_template("home.html")

# load data
mushroom_data = pd.read_csv(variables.data.mushroom.path)

# get values for dropdown form
mushroom_attributes = {
    "odors": mushroom_data['odor'].unique().tolist(),
    "gillsizes": mushroom_data['gill-size'].unique().tolist(),
    "gillcolors": mushroom_data['gill-color'].unique().tolist(),
    "stalksurfaces": mushroom_data['stalk-surface-above-ring'].unique().tolist(),
    "ringtypes": mushroom_data['ring-type'].unique().tolist(),
    "sporecolors": mushroom_data['spore-print-color'].unique().tolist()
}

# Mushrooms
@app.route("/mushroom-poison-detector", methods=["POST", "GET"])
def mushroom_page():
    # process form when submitted
    if request.method == "POST":

        # change submitted form into list and converted into a dataframe
        features = [x for x in request.form.values()]
        final = np.array(features)
        data_unseen = pd.DataFrame([final], columns=variables.columns.mushroom)

        # ensure datatype is correct
        int_columns = ["bruises"]
        data_unseen[int_columns] = data_unseen[int_columns].astype(int)

        # pass prediction into the model
        prediction = pc.predict_model(mushroom_model, data=data_unseen)
        prediction = prediction["prediction_label"].values[0]   

        # return predictions and the dropdown attributes
        return render_template("mushroom_poison_detector.html", pred=prediction, mushroom_attributes=mushroom_attributes)
    # provide dropdown attributes
    return render_template("mushroom_poison_detector.html", mushroom_attributes=mushroom_attributes)


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


# Load the unique values from the CSV files
streets_by_town_df = pd.read_csv('data/housing/street_names_by_town.csv')
flat_types_df = pd.read_csv('data/housing/flat_types.csv')
storey_range_df = pd.read_csv('data/housing/storey_range.csv')
flat_model_df = pd.read_csv('data/housing/flat_model.csv')

# Create a dictionary mapping towns to street names
streets_by_town = streets_by_town_df.set_index('town')['street_names'].apply(eval).to_dict()

# Extract unique values from the dataset
flat_types = flat_types_df['flat_type'].tolist()
storey_range = storey_range_df['storey_range'].tolist()
flat_model = flat_model_df['flat_model'].tolist()


@app.route("/house-price-prediction", methods=["POST", "GET"])
def house_page():
    if request.method == "POST":
        # Extract form data
        form_data = request.form.to_dict()
        
        # Convert the 'month' to a datetime object
        form_data['month'] = pd.to_datetime(form_data['month'])
        
        # Convert all data to the correct format and prepare for prediction
        data_unseen = pd.DataFrame([form_data], columns=variables.columns.housing)
        
        # Ensure numeric fields are converted appropriately
        data_unseen = data_unseen.astype({
            'postal_code': 'int32',
            # 'floor_area_sqm ': 'float64',
            'lease_commence_date': 'int64',
            'cbd_dist': 'float64',
            'min_dist_mrt': 'float64'
        })
        
        # Predict using the model
        prediction = pr.predict_model(house_model, data=data_unseen)
        prediction = prediction["prediction_label"].values[0]
        
        # Format the prediction value
        formatted_prediction = '${:,.0f}'.format(prediction)

        return render_template("house_price_prediction.html", pred=formatted_prediction, towns=streets_by_town.keys(), streets_by_town=streets_by_town, flat_types=flat_types, storey_range=storey_range, flat_model=flat_model)
    
    return render_template("house_price_prediction.html", streets_by_town=streets_by_town, towns=streets_by_town.keys(), flat_types=flat_types, storey_range=storey_range, flat_model=flat_model)

@app.route("/predict-house-api", methods=["POST"])
def predict_house_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = pr.predict_model(house_model, data=data_unseen)
    prediction = prediction["prediction_label"].values[0]
    return jsonify(prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(debug=False, host="0.0.0.0", port=port)
