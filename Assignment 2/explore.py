from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from utility import set_premium_price
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"

customers = pd.read_csv(DATA_DIR / "customer_data.csv")
claims = pd.read_csv(DATA_DIR / "customer_claims.csv")

my_x = customers
my_y = claims["claim"]


# Split 1: 60% train, 40% rest
x_train, x_rest, y_train, y_rest = train_test_split(
    my_x, my_y, test_size=0.4, stratify=my_y, random_state=0
)

# splits the rest into half, 20% each
x_val, x_test, y_val, y_test = train_test_split(
    x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0
)

model = LogisticRegression(max_iter=1000)



y_baseline = np.full(len(y_val), y_train.mean())



best_name = None
best_score = np.inf


final_model = HistGradientBoostingClassifier(max_depth=8, random_state=0)

final_model.fit(x_train, y_train)
p_test = final_model.predict_proba(x_test)[:,1]


p = p_test
coverage = x_test["coverage_amount"]
premium = set_premium_price(p, coverage, 0.2)
expected_profit = premium - p * coverage
ratio = expected_profit/coverage

cov = coverage.to_numpy()
profit = expected_profit.to_numpy()

BUDGET = 100000000



total_profit = profit[chosen].sum()