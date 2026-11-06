import numpy as np

#expected utility functon, formula gotten from the assignment pdf
def utility(dollars):
    Exp_u = -263.31 + 22.09 * np.log(dollars + 150000)
    return Exp_u

#gains the dollar amount from a utility value
def inverse_utility(utility):
    dollars = 
    return dollars