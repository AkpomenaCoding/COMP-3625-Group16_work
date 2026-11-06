import numpy as np

#utility functon, formula gotten from the assignment pdf
def utility(dollars):
    u = -263.31 + 22.09 * np.log(dollars + 150000)
    return u

#gains the dollar amount from a utility value
def inverse_utility(u):
    dollars = np.exp((u + 263.31)/22.09) - 150000
    return dollars

# average utility of the uninsured 
def expected_utility_uninsured(p, C):
    expected_utility = (p * utility(-C)) + ((1-p) * utility(0))
    return expected_utility

# higher premium the customer would accept
def max_premium(p, C):
    max_willing = inverse_utility(expected_utility_uninsured(p, C))
    return -max_willing

def gives_utility_benefit(premium, p, C):
    # this is for if the customer is better paying our insurance or not, would work on after work
    return