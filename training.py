"""
Python script meant to train an ML model.
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import json
from joblib import dump
from common_functions import preprocess_data


with open('config.json', 'r') as f:
    """
    Load config.json and get path variables.
    """
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


def train_model():
    """
    Function for training the model.
    """
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    df_x, df_y, encoder = preprocess_data(df, None)
    
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20)

    # Use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='ovr',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False
    )
    
    # Fit the logistic regression to your data
    model.fit(x_train, y_train)
    
    print(model.score(x_train, y_train))
    print(model.score(x_test, y_test))
    
    # Write the trained model to your workspace in a file called trainedmodel.pkl
    dump(model, os.path.join(model_path, "trainedmodel.pkl"))
    dump(encoder, os.path.join(model_path, "encoder.pkl"))


if __name__ == "__main__":
    train_model()
