from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

customers = pd.read_csv(DATA_DIR / "customer_data.csv")
claims = pd.read_csv(DATA_DIR / "customer_claims.csv")

my_x = customers
my_y = claims["claim"]

"""
print(my_x.shape)

print(my_y.shape)
print(my_y.mean())
"""

# Split 1: 60% train, 40% rest
x_train, x_rest, y_train, y_rest = train_test_split(
    my_x, my_y, test_size=0.4, stratify=my_y, random_state=0
)

x_val, x_test, y_val, y_test = train_test_split(
    x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0
)

print(x_train.shape, x_val.shape, x_test.shape)
print(y_train.mean(), y_val.mean(), y_test.mean())