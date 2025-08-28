from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    DecisionTreeClassifier,
    NaiveBayes,
    LinearSVC,
)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml import Pipeline
from datetime import datetime
from .deployment import deploy_best_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd 
from pyspark.sql.types import *
import mlflow
import mlflow.spark
import json
import os

client = mlflow.tracking.MlflowClient()
registered_model_name="TitanicClassifier"

# Base artifacts folder
artifact_base = "Artifacts"
cm_folder = os.path.join(artifact_base, "Confusion_Matrix")
report_folder = os.path.join(artifact_base, "Evaluation_Report")

# Create folders if they don't exist
os.makedirs(cm_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
from .spark_session import spark_session_creator

spark = spark_session_creator()
from .preprocess import preprocessing_pipeline
preprocessing =preprocessing_pipeline()

schema = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Survived", IntegerType(), True),
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
df=spark.read.csv(r"/root/AILabProject/data/train.csv",header=True,schema=schema)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)


train_df=train_df.withColumnRenamed('Survived','label')
test_df=test_df.withColumnRenamed('Survived','label')

# Define models
lr  = LogisticRegression(featuresCol="features", labelCol="label")
rf  = RandomForestClassifier(featuresCol="features", labelCol="label")
gbt = GBTClassifier(featuresCol="features", labelCol="label")
dt  = DecisionTreeClassifier(featuresCol="features", labelCol="label")
nb  = NaiveBayes(featuresCol="features", labelCol="label")
svc = LinearSVC(featuresCol="features", labelCol="label")


paramGrid_lr = (ParamGridBuilder()
                .addGrid(lr.regParam, [0.01, 0.1])
                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                .build())

paramGrid_rf = (ParamGridBuilder()
                .addGrid(rf.numTrees, [10, 20])
                .addGrid(rf.maxDepth, [3, 5,10])
                .build())

paramGrid_gbt = (ParamGridBuilder()
                .addGrid(gbt.maxDepth, [3, 5,10])
                .addGrid(gbt.maxIter, [10, 20, 30])
                .build())

paramGrid_dt = (ParamGridBuilder()
                .addGrid(dt.maxDepth, [2, 5, 10,20])
                .build())

paramGrid_nb = (ParamGridBuilder()
                .addGrid(nb.smoothing, [0.5, 1.0, 1.5])
                .build())

paramGrid_svc = (ParamGridBuilder()
                .addGrid(svc.regParam, [0.01, 0.1])
                .build())

preprocessing =preprocessing_pipeline()

pipelines = [
    (Pipeline(stages=[preprocessing,  lr]),  paramGrid_lr),
    (Pipeline(stages=[preprocessing,  rf]),  paramGrid_rf),
    (Pipeline(stages=[preprocessing,  gbt]), paramGrid_gbt),
    (Pipeline(stages=[preprocessing,  dt]),  paramGrid_dt),
    (Pipeline(stages=[preprocessing,  nb]),  paramGrid_nb),
    (Pipeline(stages=[preprocessing,  svc]), paramGrid_svc),
]

accuracy = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
f1_score = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")


best_model = None
best_f1 = 0
best_acc= 0
best_auc= 0

train_df=train_df.coalesce(4).cache()
test_df=test_df.coalesce(2).cache()

for pipeline, grid in pipelines:
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=grid,
                        evaluator=accuracy,
                        numFolds=3,
                        parallelism=4 )
    time=datetime.now().strftime("%d%m%Y%H%M%S")
    with mlflow.start_run(run_name=f"Experiment_{time}"):

        model = cv.fit(train_df)
        preds = model.transform(test_df)

        f1_s = f1_score.evaluate(preds)
        acc_s=accuracy.evaluate(preds)
        auc_s=auc.evaluate(preds)

        best_cv_model=model.bestModel
        bestParams = {p.name: best_cv_model.stages[-1].getOrDefault(p) 
                        for p in best_cv_model.stages[-1].extractParamMap()}
        mlflow.log_params(bestParams)
        mlflow.log_metric("Accuracy", acc_s)
        mlflow.log_metric("F1 Score", f1_s)
        mlflow.log_metric("AUC", auc_s)

        model_name=pipeline.getStages()[-1].__class__.__name__

        preds_pd = preds.select("label", "prediction").toPandas()
        cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_file = os.path.join(cm_folder,f"confusion_matrix_{model_name}.png")
        plt.savefig(cm_file)
        plt.close()
        mlflow.log_artifact(cm_file)

        report = {
            "accuracy": acc_s,
            "f1": f1_s,
            "auc": auc_s
        }
        report_file = os.path.join(report_folder, f"evaluation_report_{model_name}.json")
        with open(report_file, "w") as f:
            json.dump(report, f)
        mlflow.log_artifact(report_file)

        print(f"{pipeline.getStages()[-1].__class__.__name__} Accuracy= {acc_s:.4f}, AUC={auc_s:.4f}, F1 Score = {f1_s:.4f} ")
        
        if acc_s > best_acc:
            best_acc = acc_s
            best_f1 = f1_s
            best_auc= auc_s
            best_model = model

best_pipeline_model = best_model.bestModel
preds = best_pipeline_model.transform(test_df)

for stage in best_pipeline_model.stages:
    stage_name = stage.__class__.__name__
    print(stage_name)

model_name=best_pipeline_model.stages[-1].__class__.__name__

preds_pd = preds.select("label", "prediction").toPandas()
cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_file = os.path.join(cm_folder,f"confusion_matrix_{model_name}.png")
plt.savefig(cm_file)
plt.close()
mlflow.log_artifact(cm_file)
report = {
    "accuracy": best_acc,
    "f1": best_f1,
    "auc": best_auc
}
report_file = os.path.join(report_folder, f"evaluation_report_{model_name}.json")
with open(report_file, "w") as f:
    json.dump(report, f)
mlflow.log_artifact(report_file)

# Log params from ALL stages of the pipeline
for stage in best_pipeline_model.stages:
    stage_name = stage.__class__.__name__
    print(stage_name)
    stage_params = {}
    for p in stage.params:
        if stage.isSet(p):
            stage_params[p.name] = stage.getOrDefault(p)
        else:
            stage_params[p.name] = None

    # log with prefix so params don’t overwrite each other
    for k, v in stage_params.items():
        mlflow.log_param(f"{stage_name}.{k}", v)

mlflow.log_metric("Accuracy", best_acc)
mlflow.log_metric("F1 Score", best_f1)
mlflow.log_metric("AUC", best_auc)

mlflow.spark.log_model(
        spark_model=best_pipeline_model,
        artifact_path="spark-model",
        registered_model_name=registered_model_name,
    )

latest_versions = client.get_latest_versions(registered_model_name, stages=["None"])
new_version = max([int(v.version) for v in latest_versions])

# Move to Staging
client.transition_model_version_stage(
    name=registered_model_name,
    version=new_version,
    stage="Staging"
)
print(f"Model {registered_model_name} version {new_version} moved to Staging")
print("Accuracy Score:", best_acc)

deploy_best_model()
