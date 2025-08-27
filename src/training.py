from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    DecisionTreeClassifier,
    NaiveBayes,
    LinearSVC,
)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml import Pipeline

import mlflow
import mlflow.spark

from .spark_session import spark_session_creator

spark = spark_session_creator()
from .preprocess import preprocessing_pipeline
preprocessing =preprocessing_pipeline()

train_df=spark.read.csv(r"/root/AILabProject/data/train.csv",header=True,inferSchema=True)
test_df=spark.read.csv(r"/root/AILabProject/data/test.csv",header=True,inferSchema=True)
label=spark.read.csv(r"/root/AILabProject/data/gender_submission.csv",header=True,inferSchema=True)

test_df = test_df.join(label, on="PassengerID")
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
auc = MulticlassClassificationEvaluator(labelCol="label", metricName="auc")


best_model = None
best_acc = 0

train_df=train_df.coalesce(4).cache()
test_df=test_df.coalesce(2).cache()

for pipeline, grid in pipelines:
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=grid,
                        evaluator=f1_score,
                        numFolds=3,
                        parallelism=4 )
    
    with mlflow.start_run(run_name=f"{pipeline.getStages()[-1].__class__.__name__}"):

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

        print(f"{pipeline.getStages()[-1].__class__.__name__} F1 Score = {f1_s:.4f}")
        
        if f1_s > best_acc:
            best_acc = f1_s
            best_model = model

mlflow.spark.log_model(
            spark_model=best_model.bestModel,
            artifact_path="spark-model",
            registered_model_name="TitanicClassifier"
        )

print("\nBest model:", best_model.bestModel.stages[-1])
print("F1 Score:", best_acc)