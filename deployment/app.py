from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import *

spark = SparkSession.builder.appName("TitanicClassifierAPI").getOrCreate()

schema = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Ticket", StringType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Cabin", StringType(), True),
    StructField("Embarked", StringType(), True),
])

# -------------------------------
# Load model at startup
# -------------------------------
MODEL_PATH = "deployment/model/sparkml"  # wherever you saved it
model = PipelineModel.load(MODEL_PATH)
# Define FastAPI app

# -------------------------------
app = FastAPI(title="TitanicClassifier Inference API")

# -------------------------------
# Define input schema
# -------------------------------
# Example: replace feature1, feature2 with your actual features
class PredictionRequest(BaseModel):
    PassengerId: Optional[int] = None
    Pclass: int
    Name: Optional[str] = None
    Sex: str
    Age: Optional[float] = None
    SibSp: Optional[int] = None
    Parch: int
    Ticket: Optional[str] = None
    Fare: Optional[float] = None
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None

@app.post("/predict")
def predict(request: PredictionRequest):
    print('code entered the predict statement')
    input_data = [request.dict()]
    input_df = spark.createDataFrame(input_data, schema=schema)
    input_df.show()
    preds = model.transform(input_df)
    print('transformation is done')
    print(type(preds))
    print(dir(preds))
    print(preds.columns)
    preds.describe().show()
    preds.show()
    result_df = preds.select("prediction", "probability").toPandas()
    result_df['probability'] = result_df['probability'].apply(lambda vec: vec.tolist())
    predictions = result_df.to_dict('records')
    print(predictions)
    return {"predictions": predictions}
