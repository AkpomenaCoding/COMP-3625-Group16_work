from pathlib import Path

import pandas as pd
import numpy as np

'''
This is linear regression model that predicts the probability '''

BASE_DIR = Path(__file__).resolve().parent

# load historical customer data
customer_data = pd.read_csv(BASE_DIR / 'data' / 'customer_data.csv', index_col=None)
customer_claims = pd.read_csv(BASE_DIR / 'data' / 'customer_claims.csv', index_col=None)


# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

train_data = customer_data.sample(frac=0.8, random_state=42)
claim_train_data = customer_claims.sample(frac=0.8, random_state=42)

test_data = customer_data.drop(train_data.index)
claim_test_data = customer_claims.drop(claim_train_data.index)


print(customer_claims)


'''
Pseudocode for the function to check claim probability

- Split data into test and training sets
- Break claims into two categories: 'claim' and 'no_claim'

- Count the number of claims in each category and divide by 
the total number of claims to get the probability of each category
'''

def check_claim_probability() -> float:
    
    Y = claim_train_data['claim'].map({'no_claim': 0, 'claim': 1}).to_numpy()
    probability = Y['claim'] / len(Y)
    
    return probability

'''
Need to calculate
'''

    

