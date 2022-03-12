"""
Python script meant to score an ML model.
"""
import pandas as pd
import os
from sklearn import metrics
import logging
import sys
import json
from joblib import load
from common_functions import preprocess_data

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


with open('config.json', 'r') as f:
    """
    Load config.json and get path variables.
    """
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
output_folder_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


def score_model(production=False):
    """
    This function should take a trained model, load test data, and calculate an F1 score for the model relative to the
    test data it should write the result to the latestscore.txt file.
    """
    model = load(os.path.join(model_path, "trainedmodel.pkl"))
    encoder = load(os.path.join(model_path, "encoder.pkl"))
    
    if production:
        df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    else:
        df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    df_x, df_y, _ = preprocess_data(df, encoder)

    y_pred = model.predict(df_x)

    f1 = metrics.f1_score(df_y, y_pred)

    with open(os.path.join(model_path, "latestscore.txt"), "w") as score_file:
        score_file.write(str(f1) + "\n")
    
    return f1


if __name__ == "__main__":
    logging.info("Running Scoring!")
    score_model()
    logging.info("Artifacts output written in practicemodels/latestscore.txt")
