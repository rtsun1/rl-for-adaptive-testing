import pandas as pd
import numpy as np

item_bank_fname = 'real item bank.csv'  # (n_items, 1[a], 1[b], 1[c])
true_theta_fname = 'true theta for training.csv'  # table w/ 1 col: theta scalar. rows: subj_id (n_subj[theta])
resp_fname = 'real responses for training.csv'  # table w/ shape: (n_subj, n_items, 1[1 for right 0 for wrong])

n_items = 371
n_subj = 5000

# item bank
item_bank = pd.DataFrame({
    "a": np.random.normal(1.2, 0.25, n_items),  # disc
    "b": np.random.normal(0, 1, n_items),  # diff
    "c": np.random.normal(0.25, 0.02, n_items)  # guess
})

# theta
thetas = pd.DataFrame(np.array([
    np.random.normal(0, 1, n_subj)
])).T

# resp
resp = pd.DataFrame(np.array([
    np.random.choice([0, 1], n_subj) for 
    _ in range(n_items)
])).T

print("--Item Bank--")
print(item_bank.head())
item_bank.to_csv(item_bank_fname, header=True, index=False)

print("--Thetas--")
print(thetas.head())
thetas.to_csv(true_theta_fname, header=True, index=False)

print("--Responses--")
print(resp.head())
resp.to_csv(resp_fname, header=True, index=False)
