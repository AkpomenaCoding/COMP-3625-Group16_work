import numpy as np
import matplotlib.pyplot as plt

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