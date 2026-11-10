import pandas as pd
import numpy as np
from utility import set_premium_price, gives_utility_benefit
from selection import hill_climb, greedy_select
from matplotlib.path import Path
import pickle

BUDGET = 100_000_000
MARKUP = 0.2 # example given from the assignment doc

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "logistic_model.pkl"

with open(model_path, "rb") as file:
        saved_data = pickle.load(file)
        
model = saved_data["model"]
threshold = saved_data["threshold"]
feature_columns = saved_data["feature_columns"]
        
        
# At grading time, I will call the below function with a dataframe of new potential customers
# The dataframe I pass in will have the same columns as customer_data.csv, but will not include any of the customers in customer_data.csv
# You will not be given any of the historical customer data or claims data at grading time, 
# so you will need to save any models you train in model_dev.py and load them here.

def set_premium(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a dataframe of data on potential customers, identifies which customers to take on, and sets premiums for them.
    This function may load a saved model (previously trained using supervised learning) to save time.
    :param customer_data: a dataframe with the same columns as customer_data.csv
    :return: a DataFrame of the same lenght as customer_data, with the following columns:
            p_claim:  your estimated probability that each customer in customer_data will make a claim
            desired_customer:  a boolean column indicating which customers you want to take on
            premium:    the amount of the premium to charge each customer
    """
    X = customer_data[feature_columns]

    p_claims = model.predict_probability(customer_data)(X)[:, 1]
    # need to add utility selection function
    coverage = customer_data["coverage_amount"].to_numpy(dtype=float)

    # need to add premium selection function
    premium = set_premium_price(p_claims, coverage, MARKUP)
    profit = premium - p_claims * coverage

    eligible = gives_utility_benefit(premium, p_claims, coverage)

    desired_customer, total = greedy_select(coverage, profit, eligible, BUDGET)
    desired_customer, total, swaps = hill_climb(desired_customer, total, coverage, profit, eligible, BUDGET)

    print("Coverage shows, ", total)
    print("Expected profit, ", profit[desired_customer].sum())
    
    return pd.DataFrame({
        "p_claim": p_claims,
        "desired_customer": desired_customer,
        "premium": premium
    })
    

# TESTING SET PREMIUM FUNCTION
def main():
    customer_data = pd.read_csv(BASE_DIR / 'data' / 'customer_data.csv', index_col=None)

    result_df = set_premium(customer_data)

    print(result_df)
if __name__ == "__main__":
    main()
