import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# seed fixed here
np.random.seed(1)

# read csv 
path = "data/pisa_data.csv"
raw = pd.read_csv(path)

# randomly split raw into train, val, test
train_percent = 0.8
val_percent = (1 - train_percent) / 2

students = raw["student_id"].unique()
num_students = len(students)

train_split_idx = int(train_percent * num_students)
val_split_idx = train_split_idx + int(val_percent * num_students)

# so that rl & ml models have the same train val and test sets
indices = np.arange(num_students)
np.random.shuffle(indices)



def dataset_rl():
    '''
    Creates training data for the rl model

    Returns:
        A tuple: (train_set: pd.dataframe,
                 val_set: pd.dataframe,
                 test_set: pd.dataframe
                )
    '''

    num_q_per_student = int( len(raw["student_id"]) / len(raw["student_id"].unique()) )     # 244

    data = pd.DataFrame(np.zeros_like(raw), columns=raw.columns)
    data = pd.concat([raw.iloc[i * num_q_per_student : (i+1) * num_q_per_student] for i in indices], ignore_index=True)

    train_set = data.iloc[ : train_split_idx * num_q_per_student]
    val_set = data.iloc[train_split_idx * num_q_per_student : val_split_idx * num_q_per_student]
    test_set = data.iloc[val_split_idx * num_q_per_student : ]

    return train_set, val_set, test_set



class DatasetML(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        assert len(X) == len(Y), "Data length not match!"
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



def dataloader_ml(batch_size_train: int = 128, 
                  batch_size_val: int = 128, 
                  batch_size_test: int = 128,
                  code_ok: bool = True        # this is for testing the correctness of the code
):
    """  
    (assuming code_ok is True)

    - Divides dataset into features X and values Y (final_score), 
    - Changes X into the following format (ndim = 3), with some columns deleted:

        [ [ [ "discrimination","difficulty","response","response_time" ],       # student 1's answer for q1
            ...
            [ "discrimination","difficulty","response","response_time" ]
            ],       # student 1's answers for the whole test

            ...

        ]

    - Randomly splits data
    - Creates pytorch dataset and dataloader

    Inputs:
        batch_size_train (int)
        batch_size_val (int)
        batch_size_test (int)
        code_ok (bool): just use True when calling this function
    
    Returns:
        A tuple: (train_loader, val_loader, test_loader)    pytorch dataloaders

    Usage:
        train_loader_ml, val_loader_ml, test_loader_ml = dataloader_ml()
    """

    # extract feature-label pair
    Y = raw["final_score"].unique()
    raw_X = raw.drop(columns="final_score")
    raw_X = raw_X.sort_values(by=["student_id", "item_id"], ascending=[True, True])
    raw_X = raw_X.groupby("student_id")
    

    if not code_ok:
        col_to_delete = ["item_id"] 
    else:
        col_to_delete = ["student_id", "item_id"]       # change this if you'd like

    X = np.array([student_info.drop(columns=col_to_delete) for _, student_info in raw_X])

    # randomly split into train, val and test sets
    X, Y = X[indices], Y[indices]

    train_X, val_X, test_X = np.split(X, [train_split_idx, val_split_idx])
    train_Y, val_Y, test_Y = np.split(Y, [train_split_idx, val_split_idx])

    # create dataloaders
    train_set, val_set, test_set = DatasetML(train_X, train_Y), DatasetML(val_X, val_Y), DatasetML(test_X, test_Y)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size_train,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_set,
                              batch_size=batch_size_val,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_set,
                              batch_size=batch_size_test,
                              shuffle=False)

    if not code_ok:
        return train_set, val_set, test_set
    else:
        return train_loader, val_loader, test_loader



if __name__ == "__main__":
    # print("Testing dataloader for ml ...")
    # train_loader_ml, val_loader_ml, test_loader_ml = dataloader_ml()
    # X, Y = next(iter(train_loader_ml))
    # print(X.shape)
    # print(Y.shape)

    # print()

    # print("Testing dataset for rl ...")
    train_set_rl, val_set_rl, test_set_rl = dataset_rl()
    # print(train_set_rl)

    # print()

    print("Some further testing ...")
    train_set_ml, val_set_ml, test_set_ml = dataloader_ml(code_ok=False)

    print(len(train_set_ml), len(val_set_ml), len(test_set_ml))
    # print(len(train_set_rl) / 244, len(val_set_rl) / 244, len(test_set_rl) / 244)
    print(len(train_set_rl), len(val_set_rl), len(test_set_rl))

    idx_to_test = 50
    print(indices[idx_to_test])
    print(train_set_ml[idx_to_test])
    print(train_set_rl[idx_to_test * 244 : (idx_to_test+1) * 244])

    # print(train_set_rl.shape)
    