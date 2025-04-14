import matplotlib.pyplot as plt
import numpy as np

x = np.array(["Greedy Time Aware", "DT", "DT Time Aware", "DQN"])
y = np.array([0.35, 0.458, 0.510, 0.363])

plt.bar(x, y)
plt.xlabel("Approach")
plt.ylabel("RMSE")
plt.show()