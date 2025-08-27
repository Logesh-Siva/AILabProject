import mlflow
import mlflow.spark
from .spark_session import spark_session_creator

mlflow.set_tracking_uri("http://localhost:5000")
model_name = "TitanicClassifier"

spark=spark_session_creator()

def deploy_best_model():
    client = mlflow.tracking.MlflowClient()
    all_versions = client.get_latest_versions(name=model_name, stages=["None", "Staging", "Production"])

    best_version = None
    best_acc = -1

    for v in all_versions:
        run_id = v.run_id
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        acc = metrics.get("accuracy", None)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_version = v.version

    print(f"Best version by accuracy: {best_version} with accuracy {best_acc:.4f}")

    client.transition_model_version_stage(
        name=model_name,
        version=best_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model {model_name} version {best_version} promoted to Production")

    model_uri = f"models:/{model_name}/{best_version}"
    spark_model = mlflow.spark.load_model(model_uri)
    print(f"Model {model_name} version {best_version} loaded successfully")
