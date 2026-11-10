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

"""
print(my_x.shape)

print(my_y.shape)
print(my_y.mean())
"""

# Split 1: 60% train, 40% rest
x_train, x_rest, y_train, y_rest = train_test_split(
    my_x, my_y, test_size=0.4, stratify=my_y, random_state=0
)

# splits the rest into half, 20% each
x_val, x_test, y_val, y_test = train_test_split(
    x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0
)

"""
print(x_train.shape, x_val.shape, x_test.shape)
print(y_train.mean(), y_val.mean(), y_test.mean())
"""

model = LogisticRegression(max_iter=1000)

"""
model.fit(x_train, y_train)
p_val = model.predict_proba(x_val)[:,1]

print(p_val[:5])
print(p_val.min(), p_val.max(), p_val.mean())
"""

y_baseline = np.full(len(y_val), y_train.mean())

"""
# log loss compares our how good our predicitons were, while brier the average of our errors in predicting it.,
# the baseline is 0.335 which is the general average from my tests before. 
# y_baseline is just array with the same length as y_val filled with the mean 0.335,
# after log_loss comapres it seems that we're not to far off our baseline which I see as good. 
print(log_loss(y_val, p_val), log_loss(y_val, y_baseline))  
print(brier_score_loss(y_val, p_val), brier_score_loss(y_val, y_baseline))
"""

candidates = {
    "logistic": LogisticRegression(max_iter=1000),
    "boosting": HistGradientBoostingClassifier(random_state=0),
    "boosting3": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0),
    "boosting8": HistGradientBoostingClassifier(max_depth=8),
    "boosting10": HistGradientBoostingClassifier(max_depth=10, learning_rate=0.1, random_state=0)
}

best_name = None
best_score = np.inf

for name, model in candidates.items():
    model.fit(x_train, y_train)
    p_val = model.predict_proba(x_val)[:,1]
    curr_log_loss = log_loss(y_val, p_val)

    if (curr_log_loss < best_score):
        best_score = curr_log_loss
        best_name = name
    # print(name, curr_log_loss)

"""
print("Best: ", best_name, best_score)
"""

final_model = HistGradientBoostingClassifier(max_depth=8, random_state=0)

final_model.fit(x_train, y_train)
p_test = final_model.predict_proba(x_test)[:,1]

"""
print("----- Against P_test ------")
print(log_loss(y_test, p_test))
print(brier_score_loss(y_test, p_test))

# this baseline is for the test set. 
new_y_baseline = np.full(len(y_test), y_train.mean())
print("----- Against Baseline ------")
print(log_loss(y_test, new_y_baseline))
print(brier_score_loss(y_test, new_y_baseline))
"""

p = p_test
coverage = x_test["coverage_amount"]
premium = set_premium_price(p, coverage, 0.2)
expected_profit = premium - p * coverage
ratio = expected_profit/coverage

cov = coverage.to_numpy()
profit = expected_profit.to_numpy()

BUDGET = 100000000



total_profit = profit[chosen].sum()