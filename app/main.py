from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI()

model = mlflow.pyfunc.load_model("models:/ChurnModel/Production")

LOG_FILE = "data/production_logs.csv"

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # log incoming data
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

    pred = model.predict(df)[0]
    return {"prediction": int(pred)}