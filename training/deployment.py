import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
registered_model_name = "TitanicClassifier"
output_dir = "deployment/model"

def deploy_best_model():
    client = MlflowClient()

    # Step 1: Find the best version by accuracy
    all_versions = client.get_latest_versions(name=registered_model_name, stages=["None", "Staging", "Production"])
    best_version = None
    best_acc = -1

    for v in all_versions:
        run = mlflow.get_run(v.run_id)
        acc = run.data.metrics.get("Accuracy")  # make sure metric name matches
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_version = v.version

    if best_version is None:
        print("No model found with accuracy metric.")
        return

    print(f"Best model: version {best_version} with accuracy {best_acc:.4f}")

    # Step 2: Promote best model to Production
    client.transition_model_version_stage(
        name=registered_model_name,
        version=best_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model {registered_model_name} version {best_version} promoted to Production.")

    # Step 3: Load the Production model
    model_uri = f"models:/{registered_model_name}/Production"
    spark_model = mlflow.spark.load_model(model_uri)

    # Step 4: Save locally for deployment
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mlflow.spark.save_model(spark_model=spark_model, path=output_dir)
    print(f"Packaged Production model v{best_version} saved locally at: {output_dir}")
