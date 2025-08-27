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

import mlflow
import mlflow.spark
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
from .spark_session import spark_session_creator

spark = spark_session_creator()
from .preprocess import preprocessing_pipeline
preprocessing =preprocessing_pipeline()

df=spark.read.csv(r"/root/AILabProject/data/train.csv",header=True,inferSchema=True)
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

        model = cv.fit(train_df)   # train
        preds = model.transform(test_df)

        f1_s = f1_score.evaluate(preds)
        acc_s=accuracy.evaluate(preds)
        auc_s=auc.evaluate(preds)

        best_cv_model=model.bestModel
        bestParams = {p.name: best_cv_model.stages[-1].getOrDefault(p) 
                        for p in best_cv_model.stages[-1].extractParamMap()}
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_params(bestParams)
        mlflow.log_metric("Accuracy", acc_s)
        mlflow.log_metric("F1 Score", f1_s)
        mlflow.log_metric("AUC", auc_s)

        print(f"{pipeline.getStages()[-1].__class__.__name__} Accuracy= {acc_s:.4f}, AUC={auc_s:.4f}, F1 Score = {f1_s:.4f} ")
        
        if acc_s > best_acc:
            best_acc = acc_s
            best_f1 = f1_s
            best_auc= auc_s
            best_model = model

best_pipeline_model = best_model.bestModel
final_model = best_pipeline_model.stages[-1]

mlflow.log_metric("accuracy", best_acc)
mlflow.log_metric("auc", best_auc)
mlflow.log_metric("f1", best_f1)

import json

report = {
    "accuracy":best_acc,
    "f1": best_f1,
    "auc": best_auc
}

params = {}
for p in final_model.params:
    if final_model.isSet(p):
        params[p.name] = final_model.getOrDefault(p)
    else:
        params[p.name] = None  # or skip logging

for k, v in params.items():
    mlflow.log_param(k, v)

mlflow.spark.log_model(
        spark_model=best_pipeline_model,
        artifact_path="spark-model",
        registered_model_name="TitanicClassifier",
    )

print("Accuracy Score:", best_acc)


