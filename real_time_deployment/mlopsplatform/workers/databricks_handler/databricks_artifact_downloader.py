import mlflow
import sys
import os

databricks_host = sys.argv[1]
databricks_token = sys.argv[2]
model_name = sys.argv[3]
model_version = str(sys.argv[4])  # ensure string

# Configure environment for Databricks
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
os.environ["DATABRICKS_HOST"] = databricks_host
os.environ["DATABRICKS_TOKEN"] = databricks_token

model_uri = f"models:/{model_name}/{model_version}"
print(f"⬇️ Downloading model from {model_uri}")

try:
    mlflow.artifacts.download_artifacts(model_uri, dst_path="databricks_download")
    print("✅ Model downloaded to databricks_download/")
except Exception as e:
    print(" Failed to download model from Databricks")
    print(str(e))
    sys.exit(1)
