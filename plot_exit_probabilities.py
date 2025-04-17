import numpy as np
import matplotlib.pyplot as plt

# Example distances
distances = [5, 10, 20]  # Distances to three exits

# Compute probabilities for different randomness strengths
strengths = [0.1, 1.0, 2.0]
probabilities = {}

for strength in strengths:
    probs = 1 / (np.array(distances) ** strength)
    probs /= probs.sum()
    probabilities[strength] = probs

# Plot
markers = ["o", "s", "D"]
plt.figure(figsize=(10, 6))
for i, (strength, probs) in enumerate(probabilities.items()):
    plt.plot(distances, probs, marker=markers[i], label=rf"$\alpha = {strength}$")

plt.xlabel("Distance to Exit")
plt.ylabel("Selection Probability")
x_labels = ["Exit A\n(5 m)", "Exit B\n(10 m)", "Exit C\n(20 m)"]
plt.xticks(distances, x_labels)

# plt.title("Exit Selection Probability vs. Distance")
plt.legend()
plt.grid(False)
plt.savefig("exit_probability.pdf")
plt.show()
