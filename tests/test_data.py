import pandas as pd

def test_no_null():
    df = pd.read_csv("data/train.csv")
    assert df.isnull().sum().sum() == 0