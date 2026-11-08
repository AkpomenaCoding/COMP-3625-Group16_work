from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

customers = pd.read_csv(DATA_DIR / "customer_data.csv")
claims = pd.read_csv(DATA_DIR / "customer_claims.csv")

""" 
print(customers.shape)
print(customers.columns)
print(customers.head())

print(claims.shape)
print(claims.columns)
print(claims.head())
"""