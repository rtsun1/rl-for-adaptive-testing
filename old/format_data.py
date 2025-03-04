import pandas as pd
import numpy as np
from tqdm import tqdm

item_bank_fname = 'real item bank.csv'  # (n_items, 1[a], 1[b], 1[c])
true_theta_fname = 'true theta for training.csv'  # table w/ 1 col: theta scalar. rows: subj_id (n_subj[theta])
resp_fname = 'real responses for training.csv'  # table w/ shape: (n_subj, n_items, 1[1 for right 0 for wrong])

# n_items = 371
# n_subj = 5000

n_test_subj = 5000

# current method: just use last 5000 ids as test

raw = pd.read_csv("data/test_data.csv")
raw = raw.sort_values(by=["student_id", "item_id"])
raw = raw.reset_index(drop=True)

items = raw["item_id"].unique()
print("N Items:", items.shape[0])

n_rows_test = n_test_subj * items.shape[0]
train = raw.iloc[:-n_rows_test]  # saving the last rows for testing

ids = train["student_id"].unique()
# ensure each id has the 117 qs
print(train.shape[0] % 117)

# item bank
disc = []
diff = []
for item in items:
    # print(train[train["item_id"] == item]["discrimination"].shape)
    disc.append(train[train["item_id"] == item]["discrimination"].iloc[0])
    diff.append(train[train["item_id"] == item]["difficulty"].iloc[0])

item_bank = pd.DataFrame({
    "a": disc,  # disc
    "b": diff,  # diff
    "c": np.full(items.shape[0], 1e-9)  # guess, make it really small since our dataset currently doesn't have guessing param
})
print("--Item Bank--")
print(item_bank.dtypes)
item_bank.to_csv(item_bank_fname, header=True, index=False)

# theta
print(train.shape[0] % ids.shape[0])

abilities = []
responses = []
n_items = items.shape[0]
for i in range(ids.shape[0]):  # assuming single id is continguous
    start = i * n_items
    stop = (i + 1) * n_items
    selection = np.arange(start=start, stop=stop)
    # abilities.append(train[train["student_id"] == id]["final_score"].iloc[0])
    # responses.append(train[train["student_id"] == id]["response"].to_numpy())
    abilities.append(train.iloc[selection]["final_score"].iloc[0])
    responses.append(train.iloc[selection]["response"].to_numpy())

thetas = pd.DataFrame(np.array([
    abilities
])).T
print("--Thetas--")
print(thetas.dtypes)
print(thetas.head())
thetas.to_csv(true_theta_fname, header=True, index=False)

# resp
resp = pd.DataFrame(responses)

print("--Responses--")
print(resp.dtypes)
print(resp.head())
resp.to_csv(resp_fname, header=True, index=False)

## tests
test_resp_fname = 'real responses for testing.csv'
test_theta_fname = 'true theta for testing.csv'

test = raw.iloc[-n_rows_test:]
# print(raw.shape[0])
# print(test.shape[0])

ids = test["student_id"].unique()

# theta
abilities = []
responses = []
n_items = items.shape[0]
for i in range(ids.shape[0]):  # assuming single id is continguous
    start = i * n_items
    stop = (i + 1) * n_items
    selection = np.arange(start=start, stop=stop)
    abilities.append(test.iloc[selection]["final_score"].iloc[0])
    responses.append(test.iloc[selection]["response"].to_numpy())

test_thetas = pd.DataFrame(np.array([
    abilities
])).T

# resp
test_resp = pd.DataFrame(responses)


print("--Test Thetas--")
print(test_thetas.head())
test_thetas.to_csv(test_theta_fname, header=True, index=False)

print("--Test Responses--")
print(test_resp.head())
test_resp.to_csv(test_resp_fname, header=True, index=False)
