"""
Python script meant to deploy a trained ML model
"""
import os
import logging
import sys
import json
from shutil import copy2


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


with open('config.json', 'r') as f:
    """
    Load config.json and correct path variable
    """
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_folder_path = config["output_folder_path"]


def store_model_into_pickle():
    """
    Function for deployment
    """
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    for file in ["ingestedfiles.txt", "trainedmodel.pkl", "encoder.pkl", "latestscore.txt"]:
        if file in ["ingestedfiles.txt"]:
            source_filepath = os.path.join(output_folder_path, file)
        else:
            source_filepath = os.path.join(model_path, file)

        new_filepath = os.path.join(prod_deployment_path, file)
        print(f'Copying {source_filepath} to {new_filepath}')
        copy2(source_filepath, new_filepath)


if __name__ == '__main__':
    logging.info("Running Deployment!")
    store_model_into_pickle()
    logging.info("Artifacts output written in production_deployment/ingestedfiles.txt")
    logging.info("Artifacts output written in production_deployment/trainedmodel.pkl")
    logging.info("Artifacts output written in production_deployment/latestscore.txt")
