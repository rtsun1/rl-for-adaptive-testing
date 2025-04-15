import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir = "all_models_rmse"
greedy_baseline_rmse = np.load(os.path.join(dir, "greedy_baseline_rmse.npy"))
dt_rmse = np.load(os.path.join(dir, "dt_rmse.npy"))

time_limits = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

plt.figure(figsize=(10, 6))

plt.plot(time_limits, greedy_baseline_rmse, color="blue", label="Greedy Baseline")
plt.plot(time_limits, dt_rmse, color="orange", label="Decision Tree")

plt.xticks(time_limits)
plt.xlabel("Time Limit (min)")
plt.ylabel("RMSE")
plt.title("Performance of Different Models under Different Time Limits")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
