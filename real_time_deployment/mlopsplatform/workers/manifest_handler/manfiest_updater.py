import os
import sys
import yaml

# --- Arguments ---
model_name = sys.argv[1]
deployment_env = sys.argv[2]
model_version = sys.argv[3]

trained_on_feature_table = ""

model_uri = f"azureblob:model-artifacts/{model_name}/{deployment_env}/{model_version}"
MODEL_SERVER = "SKLEARN_CUSTOM_SERVER"

MODEL_PATH = "mlopsplatform/models/"
DEFAULT_MANIFEST_PATH = "manifest/default_manifest.yaml"  # <-- corrected spelling

# --- Load default manifest ---
def get_manifest():
    with open(DEFAULT_MANIFEST_PATH, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.SafeLoader)

    data['metadata']['name'] = model_name
    data['metadata']['namespace'] = deployment_env
    data['spec']['predictors'][0]['graph']['modelUri'] = model_uri
    data['spec']['predictors'][0]['graph']['name'] = model_name + "-graph"
    data['spec']['predictors'][0]['graph']['implementation'] = MODEL_SERVER

    return data

# --- Create or update manifest ---
def create_new_model(modelname, modelenv, modeluri, modelversion):
    tempmodelpath = os.path.join(MODEL_PATH, modelname)
    os.makedirs(tempmodelpath, exist_ok=True)

    tempmodelpath = os.path.join(tempmodelpath, modelenv)
    os.makedirs(tempmodelpath, exist_ok=True)

    manifest_file = os.path.join(tempmodelpath, "manifest.yaml")

    if not os.path.exists(manifest_file):
        # Create new manifest
        manifest = get_manifest()
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
    else:
        # Update existing manifest
        with open(manifest_file, 'r') as stream:
            data = yaml.load(stream, Loader=yaml.SafeLoader)

        data['spec']['predictors'][0]['graph']['modelUri'] = modeluri

        with open(manifest_file, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

# --- Run ---
create_new_model(model_name, deployment_env, model_uri, model_version)
print(f"✅ Manifest created/updated for model '{model_name}' in environment '{deployment_env}'")
