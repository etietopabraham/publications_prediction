# publications_prediction
We are presented with a dataset of one popular social network that includes more than 8.5 million records with meta-information of publications over 13 months (January 2019 to February 2020). Using this data, we will need to predict the number of publications in each 250x250 meter polygon for each hour 4 weeks (28 days) ahead of the last publication in the training set.

# Let's Begin the Dance
1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update entity
5. Update configuration manager in src/config
6. Update components
7. Update pipeline
8. Update main ochestration
9. Update app.py

# How to Run
Clone the repository

```bash
git clone https://github.com/etietopabraham/publications_prediction.git
```

Create a conda environment after opening the repository
```
conda create --name predict_publications_env python=3.8 -y
conda activate predict_publications_env 

```

Conda install pip and export PATH
```
conda install pip
export PATH=/Users/macbookpro/miniconda3/envs/predict_publications_env/bin:$PATH
```

Install the requirements
```
pip install -r requirements.txt
```

# MLflow
Documentation: https://mlflow.org/docs/latest/index.html

# Signup dagshub
[dagshub](https://dagshub.com/)

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/etietopabraham/publications_prediction.mlflow
export MLFLOW_TRACKING_USERNAME=etietopabraham 
export MLFLOW_TRACKING_PASSWORD=324bb2aaa6fc82dbfce509eac2ce2cd6a016a869

```