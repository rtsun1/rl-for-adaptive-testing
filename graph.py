import os
import numpy as np
import matplotlib.pyplot as plt

time_limits = range(1, 31, 3)

dir = "all_models_rmse"
greedy_baseline_rmse = np.load(os.path.join(dir, "greedy_baseline_rmse.npy"))
dt_rmse = np.load(os.path.join(dir, "dt_rmse.npy"))

plt.figure(figsize=(10, 6))

# change below
plt.plot(time_limits, greedy_baseline_rmse, color="blue", label="Greedy Baseline")
plt.plot(time_limits, dt_rmse, color="orange", label="Decision Tree")
# end

plt.xticks(time_limits)
plt.xlabel("Time Limit (min)")
plt.ylabel("RMSE")
plt.title("RMSE of Different Models under Different Time Contstraints")
plt.legend()
plt.grid(True, linestyle='-', alpha=0.6)

plt.show()
