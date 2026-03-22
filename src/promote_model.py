from mlflow.tracking import MlflowClient

client = MlflowClient()

latest_versions = client.get_latest_versions("ChurnModel")
version = latest_versions[0].version

client.transition_model_version_stage(
    name="ChurnModel",
    version=version,
    stage="Production"
)

print("Promoted to Production")