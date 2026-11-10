import numpy as np

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
                if chosen[k] or not eligible[k]:
                    continue
                if total - cov[j] + cov[k] > budget:
                    continue
                if profit[k] <= profit[j]:
                    continue
                if best_k == -1 or profit[k] > profit[best_k]:
                    best_k = k

            if best_k != -1:
                chosen[j] = False
                chosen[best_k] = True
                total = total - cov[j] + cov[best_k]
                swaps += 1
                improved = True

        return chosen, total, swaps