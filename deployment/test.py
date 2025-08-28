import requests
import pandas as pd
from training.spark_session import spark_session_creator
url = "http://localhost:8000/predict"

test_df=pd.read_csv(r"/root/AILabProject/data/test.csv")
test_df = test_df.where(pd.notnull(test_df), None)

predictions = []

for _, row in test_df.iterrows():
    payload = row.to_dict()
    response = requests.post(url, json=payload)
    predictions.append(response.json()["predictions"][0])