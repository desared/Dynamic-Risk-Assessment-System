"""
Python script meant to ingest new data.
"""
import pandas as pd
import glob
import logging
import json
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


with open("config.json", "r") as f:
    """
    Load config.json and get input and output paths.
    """
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


def merge_multiple_dataframe():
    """
    Check for datasets, compile them together, and write to an output file.
    """
    csv_files = glob.glob("%s/*.csv" % input_folder_path)

    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)

    df.drop_duplicates(inplace=True)

    df.to_csv("%s/finaldata.csv" % output_folder_path, index=False)

    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as report_file:
        for line in csv_files:
            report_file.write(line + "\n")


if __name__ == "__main__":
    logging.info("Running Ingestion!")
    merge_multiple_dataframe()
    logging.info("Artifacts output written in ingesteddata/finaldata.csv")
    logging.info("Artifacts output written in ingesteddata/ingestedfiles.csv")
