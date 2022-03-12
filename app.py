"""
Python script meant to contain API endpoints.
"""
from flask import Flask, request
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_data, outdated_packages_list, execution_time
from scoring import score_model

# Set up variables for use in our script.
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():        
    """
    Prediction Endpoint.
    """
    # Call the prediction function you created in Step 3
    dataset_path = request.json.get('dataset_path')
    y_pred, _ = model_predictions(dataset_path)
    return str(y_pred)


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    """
    Scoring Endpoint.
    """
    # Check the score of the deployed model.
    score = score_model()
    return str(score)


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Statistics Endpoint.
    """
    # Check means, medians, and modes for each column.
    summary = dataframe_summary()
    return str(summary)


@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """
    Diagnostics Endpoint.
    """
    et = execution_time()
    md = missing_data()
    op = outdated_packages_list()     
    return str("execution_time:" + et + "\nmissing_data;"+ md + "\noutdated_packages:" + op)


if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
