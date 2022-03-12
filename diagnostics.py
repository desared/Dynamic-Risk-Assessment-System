"""
Python script meant to measure model and data diagnostics.
"""
import pandas as pd
import numpy as np
import timeit
import os
import json
from joblib import load
from scipy.sparse import data
from common_functions import preprocess_data
import subprocess
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


with open('config.json', 'r') as f:
    """
    Load config.json and correct path variable.
    """
    config = json.load(f) 

model_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path']) 


def model_predictions(dataset_path):
    """
    Function to get model predictions.
    """
    # read the deployed model and a test dataset, calculate predictions
    model = load(os.path.join(model_path, "trainedmodel.pkl"))
    encoder = load(os.path.join(model_path, "encoder.pkl"))
    
    if dataset_path is None:
        dataset_path = "testdata.csv"
    df = pd.read_csv(os.path.join(test_data_path, dataset_path))

    df_x, df_y, _ = preprocess_data(df, encoder)

    y_pred = model.predict(df_x)

    return y_pred, df_y


def dataframe_summary():
    """
    Function to get summary statistics.
    """
    # calculate summary statistics here
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    numeric_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
        ]
    
    result = []
    for column in numeric_columns:
        result.append([column, "mean", df[column].mean()])
        result.append([column, "median", df[column].median()])
        result.append([column, "standard deviation", df[column].std()])
    
    return result


def missing_data():
    """
    Function to deal with missing data.
    """
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    result = []
    for column in df.columns:
        count_na = df[column].isna().sum()
        count_not_na = df[column].count()
        count_total = count_not_na + count_na

        result.append([column, str(int(count_na/count_total*100))+"%"])
    
    return str(result)


def execution_time():
    """
    Function to get timings.
    """
    # calculate timing of training.py and ingestion.py
    result = []
    for procedure in ["training.py" , "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system('python3 %s' % procedure)
        timing=timeit.default_timer() - starttime
        result.append([procedure, timing])
 
    return str(result)


def outdated_packages_list():
    """
    Function to check dependencies.
    """
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(outdated_packages)


if __name__ == '__main__':
    logging.info("Running diagnostics!")
    model_predictions(None)
    execution_time()
    dataframe_summary()
    missing_data()
    outdated_packages_list()
