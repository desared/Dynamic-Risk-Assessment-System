# Dynamic Risk Assessment System
Project for [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) by Udacity.

## Description
This project is part of Unit 5

Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. Your industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, you need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.
## Prerequisites
- Python 3 required
- Linux environment may be needed within windows through WSL

## Dependencies
This project dependencies is available in the ```requirements.txt``` file.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies from the ```requirements.txt```. 
Its recommended to install it in a separate virtual environment.

```bash
pip install -r requirements.txt
```

or

```bash
sudo apt install pipenv
pipenv shell
pipenv install
```

## Steps Overview
1. **Data ingestion:** Automatically check if new data that can be used for model training. Compile all training data to a training dataset and save it to folder. 
2. **Training, scoring, and deploying:** Write scripts that train an ML model that predicts attrition risk, and score the model. Saves the model and the scoring metrics.
3. **Diagnostics:** Determine and save summary statistics related to a dataset. Time the performance of some functions. Check for dependency changes and package updates.
4. **Reporting:** Automatically generate plots and PDF document that report on model metrics and diagnostics. Provide an API endpoint that can return model predictions and metrics.
5. **Process Automation:** Create a script and cron job that automatically run all previous steps at regular intervals.

## Usage

### 1- Edit config.json file to use practice data

```bash
"input_folder_path": "practicedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "practicemodels", 
"prod_deployment_path": "production_deployment"
```

### 2- Run data ingestion
```python
python ingestion.py
```
Artifacts output:
```
ingesteddata/finaldata.csv
ingesteddata/ingestedfiles.txt
```

### 3- Model training
```python
python training.py
```
Artifacts output:
```
practicemodels/trainedmodel.pkl
practicemodels/encoder.pkl
```

###  4- Model scoring 
```python
python scoring.py
```
Artifacts output: 
```
practicemodels/latestscore.txt
``` 

### 5- Model deployment

```python
python deployment.py
```
Artifacts output:
```
production_deployment/ingestedfiles.txt
production_deployment/trainedmodel.pkl
production_deployment/latestscore.txt
``` 

### 6- Run diagnostics
```python
python diagnostics.py
```

### 7- Run reporting
```python
python reporting.py
```
Artifacts output:
```
practicemodels/confusionmatrix.png
```

### 8- Run Flask App
```python
python app.py
```

### 9- Run API endpoints
```python
python apicalls.py
```
Artifacts output:
```
practicemodels/apireturns.txt
```

### 11- Edit config.json file to use production data

```bash
"input_folder_path": "sourcedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "models", 
"prod_deployment_path": "production_deployment"
```

### 10- Full process automation
```python
python fullprocess.py
```
### 11- Cron job

Start cron service
```bash
sudo service cron start
```

Edit crontab file
```bash
sudo crontab -e
```
   - Select **option 3** to edit file using vim text editor
   - Press **i** to insert a cron job
   - Write the cron job in ```cronjob.txt``` which runs ```fullprocces.py``` every 10 mins
   - Save after editing, press **esc key**, then type **:wq** and press enter
  
View crontab file
```bash
sudo crontab -l
```
