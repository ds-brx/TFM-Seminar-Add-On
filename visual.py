import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

# Simulated example: Replace this with loading your Free Light Chain dataset
# For example:
data = pd.read_csv("free_light_chain_mortality.csv")

# Let's say you're using this feature:
feature_col = "creatinine"
time_col = "sample.yr"

# Sort by time if not already
data = data.sort_values(time_col)

# Convert feature to numpy array
signal = data[feature_col].values.reshape(-1, 1)

# Run PELT to detect shifts
algo = rpt.Pelt(model="l2").fit(signal)
change_points = algo.predict(pen=10)

# Plot the feature over time with change points
plt.figure(figsize=(12, 5))
plt.plot(data[time_col], signal, label=feature_col, color='black')
for cp in change_points[:-1]:  # exclude final dummy breakpoint
    plt.axvline(x=data[time_col].iloc[cp], color='red', linestyle='--', label='Detected Shift' if cp == change_points[0] else "")

plt.title(f"Shift Detection in '{feature_col}' using PELT")
plt.xlabel(time_col)
plt.ylabel(feature_col)
plt.legend()
plt.tight_layout()
plt.show()
