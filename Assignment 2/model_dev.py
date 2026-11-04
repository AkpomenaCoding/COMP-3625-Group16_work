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


print(combine_data)


'''
Pseudocode for the function to check claim probability

- Split data into test and training sets
- Break claims into two categories: 'claim' and 'no_claim'

- Count the number of claims in each category and divide by 
the total number of claims to get the probability of each category
'''

def check_claim_probability() -> float:
    
    Y = train_data['claim'].map({'no_claim': 0, 'claim': 1}).to_numpy()
    probability = Y['claim'] / len(Y)
    
    return probability

'''
Need to calculate
'''

    

