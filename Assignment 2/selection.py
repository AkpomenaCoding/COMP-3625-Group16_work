import numpy as np

BUDGET = 100_000_000
MARKUP = 0.2 # example given from the assignment doc

def greedys_select(cov, profit, eligible, budget):
    order = np.argsort(-(profit/cov))
    chosen = np.zeros(len(cov), dtype=bool)
    total = 0

    i = 0
    while(i < len(cov)):
        c = order[i]
    
        if eligible[c] and (total + cov[c]) <= budget:
            chosen[c] = True
            total = total + cov[c]

        i = i + 1

def hill_climb(chosen, total, cov, profit, eligible, budget):
    improved = True
    swaps = 0

    while improved:
        improved = False

        for j in range(len(cov)):
            if not chosen[j]:
                continue

            best_k = -1

            for k in range(len(cov)):
                if chosen[k] or not eligible[k]: # if k already in the set of chosen then disregard
                    continue
                if total - cov[j] + cov[k] > budget: # if new coverage plus total is over budget, disregard
                    continue
                if profit[k] <= profit[j]: # if the profit of k is less than the current profit we have then disregard
                    continue
                if best_k == -1 or profit[k] > profit[best_k]: # sets the new best k
                    best_k = k


            if best_k != -1: 
                # this section swaps the chosen j with a good replacement then recalculates the total
                chosen[j] = False
                chosen[best_k] = True
                total = total - cov[j] + cov[best_k]
                swaps += 1
                improved = True

        return chosen, total, swaps
    # while testing with the intial data set provided, there was no good replacement with zero swaps.