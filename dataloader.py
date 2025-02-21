import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader


def split_data(train_percent: float = 0.8):
    """
    Reads csv file and determines where to split the dataset (into train, val and test set)

    Returns:
        raw (pd dataframe): raw data read from the csv file
        train_split_idx (int): data[:train_split_idx] is the train data
        val_split_idx (int): data[train_split_idx: val_split_idx] is the val data
    """

    # read csv 
    path = "data/pisa_data.csv"         # github wouldn't let me push this dataset which is too big 
    raw = pd.read_csv(path)

    # split data - train, val, test
    students = raw["student_id"].unique()
    num_students = len(students)

    val_percent = (1 - train_percent) / 2

    train_split_idx = int(train_percent * num_students)
    val_split_idx = train_split_idx + int(val_percent * num_students)

    return raw, train_split_idx, val_split_idx


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


def dataloader_ml(batch_size_train=128, batch_size_val=128, batch_size_test=128):
    """
    Based on the output from split_data(),
        - Divides dataset into features X and values Y (final_score), 
        - Changes X into the following format (ndim = 3), with some columns deleted:

            [ [ [ "discrimination","difficulty","response","response_time" ],       # student 1's answer for q1
                ...
                [ "discrimination","difficulty","response","response_time" ]
              ],       # student 1's answers for the whole test

              ...

            ]

        - Randomly splits data based on the output from split_data()
        - Creates pytorch dataset and dataloader
    
    Returns:
        A tuple: (train_loader, val_loader, test_loader)
    """

    raw, train_split_idx, val_split_idx = split_data()

    # extract feature-label pair
    Y = raw["final_score"].unique()
    raw_X = raw.drop(columns="final_score")
    raw_X = raw_X.groupby("student_id")

    col_to_delete = ["student_id", "item_id"]      # change this if you want to keep these columns
    X = np.array([student_info.drop(columns=col_to_delete) for _, student_info in raw_X])

    # randomly split into train, val and test sets
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]

    train_X, val_X, test_X = np.split(X, [train_split_idx, val_split_idx])
    train_Y, val_Y, test_Y = np.split(Y, [train_split_idx, val_split_idx])

    # create dataloaders
    train_set = DatasetML(train_X, train_Y)
    val_set = DatasetML(val_X, val_Y)
    test_set = DatasetML(test_X, test_Y)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size_train,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_set,
                              batch_size=batch_size_val,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_set,
                              batch_size=batch_size_test,
                              shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = dataloader_ml()
    X, Y = next(iter(train_loader))
    print(X.shape)
    print(Y.shape)
