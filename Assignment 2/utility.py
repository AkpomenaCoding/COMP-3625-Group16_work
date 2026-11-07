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

def gives_utility_benefit(premium, p, C) -> bool:
    # this is for if the customer is better paying our insurance or not, would work on after work
    insured_utility = utility(-premium)
    uninsured = expected_utility_uninsured(p, C)

    #returns if true if the customer would find it better to just pay our insurance
    return insured_utility > uninsured 

def set_premium_price(p, C, markup):
    expected_cost = p * C

    #the price we want to offer
    wanted_price = expected_cost * (1 + markup)

    # guarding rail for the insurance
    safe_cap = expected_cost + (0.95 * (max_premium(p, C) - expected_cost))
    
    #name can be ignored 
    my_return = np.minimum(wanted_price, safe_cap)
    return my_return

#print(set_premium_price(0.1, 50000, 0.2))

#print(gives_utility_benefit(set_premium_price(0.1, 50000, 0.2), 0.1, 50000))