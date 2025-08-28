import mlflow
from mlflow.tracking import MlflowClient
import os
import shutil

# --- Configuration ---
# Set this to the name of the registered model you want to delete
model_name_to_delete = "TitanicClassifier" 

# Set the tracking URI to your local server
# This is the default. Change it if you run mlflow on a different port.
mlflow.set_tracking_uri("http://localhost:5000")

# --- Script Logic ---
client = MlflowClient()

try:
    print(f"Attempting to delete registered model: '{model_name_to_delete}'")
    
    # 1. Find all versions of the model
    versions = client.search_model_versions(f"name='{model_name_to_delete}'")
    
    if not versions:
        print(f"No versions found for model '{model_name_to_delete}'. It may have already been deleted or never had versions.")
    
    # 2. Delete each version. This also deletes the associated artifact files.
    for v in versions:
        print(f"- Deleting version {v.version} of model {v.name}...")
        # Transition the stage to "Archived" before deleting
        client.transition_model_version_stage(
            name=v.name,
            version=v.version,
            stage="Archived"
        )
        # Now delete the model version
        client.delete_model_version(name=v.name, version=v.version)
        print(f"  Version {v.version} deleted.")
        
    # 3. Delete the (now empty) registered model
    client.delete_registered_model(name=model_name_to_delete)
    
    print(f"\n✅ Successfully deleted registered model '{model_name_to_delete}' and all its versions.")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("Please check if the model name is correct and if the MLflow server is running.")