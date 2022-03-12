"""
Script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed.
"""
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os


with open("config.json", "r") as f:
    """
    Load config.json and correct path variable.
    """
    config = json.load(f)

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path']) 

ingested_files =[]
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as report_file:
    """
    Check and read new data. Read ingestedfiles.txt.
    """
    for line in report_file:
        ingested_files.append(line.rstrip())


new_f1_scores = False
for filename in os.listdir(input_folder_path):
    """
    Determine whether the source data folder has files that aren't listed in ingestedfiles.txt. 
    """
    if input_folder_path + "/" + filename not in ingested_files:
        new_f1_scores = True

# Deciding whether to proceed, part 1.
if not new_f1_scores:
    """
    If you found new data, you should proceed. otherwise, do end the process here.
    """
    print("No new ingested data, exiting")
    exit(0)


# Checking for model drift.
# Check whether the score from the deployed model is different from the score from the model that uses the newest
# ingested data.
ingestion.merge_multiple_dataframe()
scoring.score_model(production=True)

with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as report_file:
    old_f1_score = float(report_file.read())

with open(os.path.join(model_path, "latestscore.txt"), "r") as report_file:
    new_f1_score = float(report_file.read())


# Deciding whether to proceed, part 2.
if new_f1_score >= old_f1_score:
    """
    If you found model drift, you should proceed. otherwise, do end the process here.
    """
    print(
        "Actual F1 (%s) is better/equal than old F1 (%s), no drift detected -> exiting" % (new_f1_score, old_f1_score)
    )
    exit(0)

print("Actual F1 (%s) is WORSE than old F1 (%s), drift detected -> training model" % (new_f1_score, old_f1_score)) 
training.train_model()

# Re-deployment.
# If you found evidence for model drift, re-run the deployment.py script.
deployment.store_model_into_pickle()

# Diagnostics and reporting.
# Run diagnostics.py and reporting.py for the re-deployed model.
diagnostics.model_predictions(None)
diagnostics.execution_time()
diagnostics.dataframe_summary()
diagnostics.missing_data()
diagnostics.outdated_packages_list()
reporting.score_model()
