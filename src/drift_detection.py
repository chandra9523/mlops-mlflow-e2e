import pandas as pd
from scipy.stats import ks_2samp
import sys

train = pd.read_csv("data/train.csv")
new = pd.read_csv("data/new_data.csv")

drift = False

for col in train.columns[:-1]:
    _, p = ks_2samp(train[col], new[col])
    
    if p < 0.05:
        print(f"Drift in {col}")
        drift = True

if drift:
    sys.exit(1)