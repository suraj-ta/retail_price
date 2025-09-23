import requests
import sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Args
username = sys.argv[1]
password = sys.argv[2]
argocd_url = sys.argv[3]  # should include scheme (http:// or https://)
application_name = sys.argv[4]
env = sys.argv[5]                      # for repo path
namespace = sys.argv[6] if len(sys.argv) > 6 else "default"  # for K8s namespace

repo_url = "https://dev.azure.com/tamlopsplatform/mlopsplatform/_git/mlopsplatform"

# 1. Login
login_url = f"{argocd_url}/api/v1/session"
res = requests.post(
    login_url,
    json={"username": username, "password": password},
    headers={'Content-Type': "application/json", 'Accept': "application/json"},
    verify=False
)

if res.status_code != 200:
    print("❌ Failed to login to ArgoCD")
    print(res.text)
    sys.exit(1)

token = res.json().get('token')

# 2. Define Application manifest
create_application_body = {
    "metadata": {"name": application_name},
    "spec": {
        "project": "default",
        "destination": {
            "namespace": namespace,
            "server": "https://kubernetes.default.svc"
        },
        "source": {
            "repoURL": repo_url,
            "path": f"mlopsplatform/models/{application_name}/{env}",
            "targetRevision": "master"
        },
        "syncPolicy": {
            "automated": {"prune": True, "selfHeal": True}
        }
    }
}

headers = {
    'Content-Type': "application/json",
    'Accept': "application/json",
    "Authorization": f"Bearer {token}"
}

# 3. Create/Update Application
create_url = f"{argocd_url}/api/v1/applications"
res = requests.post(create_url, json=create_application_body, headers=headers, verify=False, params={'upsert': True})

if res.status_code not in (200, 201):
    print("❌ Failed to create ArgoCD application")
    print("Failure logs:", res.text)
    sys.exit(1)

# 4. Trigger Sync
sync_url = f"{argocd_url}/api/v1/applications/{application_name}/sync"
res = requests.post(sync_url, headers=headers, verify=False)

if res.status_code not in (200, 201):
    print("❌ Failed to sync ArgoCD application")
    print("Failure reason:", res.text)
    sys.exit(1)

print(f"✅ Successfully created and synced ArgoCD application: {application_name}")
