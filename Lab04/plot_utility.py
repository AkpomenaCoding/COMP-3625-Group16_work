import numpy as np
import matplotlib.pyplot as plt

p_payout = [0.6, 0.5, 0.9, 0.4, 0.1, 0.05]
payout = [80000, 20000, 100000, 40000, 10000, 90000]

# Utility parameters for each client's two investments
owners = ["Client A", "Client A", "Client B",
          "Client B", "Client C", "Client C"]

params = [
    (0.2, 1000),
    (0.2, 1000),
    (0.14, 200),
    (0.14, 200),
    (1.43, 100000),
    (1.43, 100000)
]

Exp_utility = []
Cert_equivalent = []

print("Expected Utility and Certainty Equivalent for Each Investment \n")
print(f"{'Investment':<12}{'Owner':<12}{'Probability':>12}"
      f"{'Payout ($)':>15}{'Expected Utility':>20}"
      f"{'Cert. Equivalent ($)':>25}")

print("-" * 95)

for i in range(6):
    a, s = params[i]
    p = p_payout[i]
    V = payout[i]

    EU = p * a * np.log(V / s + 1)
    CE = s * (np.exp(EU / a) - 1)

    Exp_utility.append(EU)
    Cert_equivalent.append(CE)

    print(f"{i+1:<12}{owners[i]:<12}{p:>12.0%}"
          f"{V:>14,.2f}{EU:>20.4f}"
          f"{CE:>19,.2f}\n")

print("Task 3 \n")
p = 0.9
V = 100000

# Client B (seller)
a_B, s_B = 0.14, 200

EU_B = p * a_B * np.log(V / s_B + 1)
CE_B = s_B * (np.exp(EU_B / a_B) - 1)

# Client C (buyer)
a_C, s_C = 1.43, 100000

EU_C = p * a_C * np.log(V / s_C + 1)
CE_C = s_C * (np.exp(EU_C / a_C) - 1)

# Maximum price Client C would pay
price = CE_C

# Utility gains at the maximum price
gain_B = a_B * np.log(price / s_B + 1) - EU_B
gain_C = EU_C - a_C * np.log(price / s_C + 1)

print(f"Client B expected utility: {EU_B:.4f}")
print(f"Client B certainty equivalent: ${CE_B:,.2f}")
print(f"Client C expected utility: {EU_C:.4f}")
print(f"Client C certainty equivalent: ${CE_C:,.2f}")
print(f"Maximum selling price: ${price:,.2f}")
print(f"Client B utility gain: {gain_B:.4f}")
print(f"Client C utility gain: {gain_C:.4f}")


# Generate dollar amounts from $0 to $100,000
money = np.arange(0, 100001, 1)

# Calculate each client's utility
utility_A = 0.2 * np.log(money / 1000 + 1)
utility_B = 0.14 * np.log(money / 200 + 1)
utility_C = 1.43 * np.log(money / 100000 + 1)

# Plot all three utility functions
plt.plot(money, utility_A, label="Client A")
plt.plot(money, utility_B, label="Client B")
plt.plot(money, utility_C, label="Client C")

plt.xlabel("Money ($)")
plt.ylabel("Utility")
plt.title("Utility Functions for Three Clients")
plt.legend()
plt.grid(True)
plt.show()
