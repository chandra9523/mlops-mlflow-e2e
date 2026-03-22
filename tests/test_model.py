import mlflow.pyfunc

def test_model_load():
    model = mlflow.pyfunc.load_model("models:/ChurnModel/Production")
    assert model is not None