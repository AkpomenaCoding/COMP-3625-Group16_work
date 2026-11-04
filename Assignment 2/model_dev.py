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

combine_data = pd.concat([customer_data, customer_claims],axis=1)

train_data = combine_data.sample(frac=0.6, random_state=42)

test_data = combine_data.drop(train_data.index)

train_data_Y = train_data['claim'].map({'no_claim': 0, 'claim': 1}).to_numpy()

print(combine_data)


'''
Pseudocode for the function to check claim probability
'''

'''
    Calculate the weight of the claim category
'''
def calc_weight(w: np.array) -> float:

    weight_length = len(w)
    weights = train_data @ w[0:(weight_length-1)] + w[weight_length-1]
    
    return weights


def check_claim_probability(w: np.array) -> float:
    
    weight = calc_weight(w)
    probability = 1 / (1 + np.exp(-weight))
    
    return probability


    

