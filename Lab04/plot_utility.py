import numpy as np
import matplotlib.pyplot as plt

p_payout = [0.6, 0.5, 0.9, 0.4, 0.1, 0.05]
payout = [80000, 20000, 100000, 40000, 10000, 90000]

Exp_utility = []
Cert_equivalent = []

i = 0

while (i < 2):
    a, s, p, V = 0.2, 1000, p_payout[i], payout[i]
    EU = p * a * np.log(V/s + 1)
    Exp_utility.append(EU)
    CE = s * (np.exp(EU/a) - 1)
    Cert_equivalent.append(CE)
    i = i + 1

while(i < 4):
    a, s, p, V = 0.14, 200, p_payout[i], payout[i]
    EU = p * a * np.log(V/s + 1)
    Exp_utility.append(EU)
    CE = s * (np.exp(EU/a) - 1)
    Cert_equivalent.append(CE)
    i = i + 1

while(i < 6):
    a, s, p, V = 1.43, 100000, p_payout[i], payout[i]
    EU = p * a * np.log(V/s + 1)
    Exp_utility.append(EU)
    CE = s * (np.exp(EU/a) - 1)
    Cert_equivalent.append(CE)
    i = i + 1

print(Exp_utility)
print(Cert_equivalent)

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
