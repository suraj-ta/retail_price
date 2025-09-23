from azure.storage.blob import BlobServiceClient
import sys
import os

azureblobaccountname = sys.argv[1]
azureblobaccountkey = sys.argv[2]
model_name = sys.argv[3]
model_env = sys.argv[4]
model_version = str(sys.argv[5])

connect_str = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={azureblobaccountname};"
    f"AccountKey={azureblobaccountkey};"
    f"EndpointSuffix=core.windows.net"
)

blob_service_client = BlobServiceClient.from_connection_string(connect_str)

container_name = "model-artifacts"
container_client = blob_service_client.get_container_client(container_name)
try:
    container_client.create_container()
except Exception:
    # container already exists
    pass

local_dir = "databricks_download"

print(f"⬆️ Uploading model artifacts from {local_dir} to container '{container_name}'")

for root, _, files in os.walk(local_dir):
    for file in files:
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, local_dir)
        blob_path = f"{model_name}/{model_env}/{model_version}/{rel_path}"

        print(f"  → Uploading {rel_path} → {blob_path}")
        with open(full_path, "rb") as data:
            container_client.upload_blob(name=blob_path, data=data, overwrite=True)

print("✅ All model artifacts uploaded successfully!")
