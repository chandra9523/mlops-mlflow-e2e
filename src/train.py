import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

mlflow.set_experiment("churn_prediction")

df = pd.read_csv("data/train.csv")

X = df.drop("target", axis=1)
y = df["target"]

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    signature = infer_signature(X, y)

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name="ChurnModel"
    )

print("Model logged to MLflow")