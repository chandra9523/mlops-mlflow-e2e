import pandas as pd
import mlflow.pyfunc
from sklearn.metrics import accuracy_score
import sys

model = mlflow.pyfunc.load_model("models:/ChurnModel/Production")

df = pd.read_csv("data/new_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

preds = model.predict(X)

acc = accuracy_score(y, preds)

print("Accuracy:", acc)

if acc < 0.7:
    print("Performance degraded")
    sys.exit(1)