# look up table for evaluation
import time
import os
import pickle
import numpy as np
import pandas as pd
from dataloader import dataset_rl as dataset_dt
from dt import DTDiscrete, QuestionNode

import torch
import torch.torch_version
from ability_level_model import RegressionModel

ABILITY_MIN = -4
ABILITY_MAX = 4

MAX_QUESTIONS = 15
MAX_TIME = 900

NUM_QUESTIONS = 244


# prepare data for evaluation
def generate_trajectories(model, useful_tables, device):
    '''
    Generate and save the trajectory of questions answered by a student, each trajectory has the shape (244, 4) and looks like 
    [ [ "discrimination", "difficulty", "response", "response_time" ],      # q110      note that the response time should be the one in the val set not the avg time saved in the dt
      [ "discrimination", "difficulty", "response", "response_time" ],      # q231
        ...
      [ -1, -1, -1, -1]    # invalid q
    ]

    Parameter:
        - model 
        - student_id (int)
    '''

    [discrimination_table, difficulty_table, response_table, time_table, stu_ids] = [useful_tables[i] for i in range(5)]

    N = len(response_table)

    trajectories = torch.full(size=(N, NUM_QUESTIONS, 4), fill_value=-1.0).to(device)

    for i in range(N):
        time_remain = MAX_TIME
        num_qs = 0
        current_q = model.root_q
        j = 0

        while current_q is not None:
            current_q_id = current_q.item_id

            current_q_discrimination = discrimination_table.loc[current_q_id]
            current_q_difficulty = difficulty_table.loc[current_q_id]
            current_q_response = response_table.loc[stu_ids[i], current_q_id]
            current_q_response_time = time_table.loc[stu_ids[i], current_q_id]

            num_qs += 1
            time_remain -= current_q_response_time
            if time_remain < 0 or num_qs > MAX_QUESTIONS:
                break
            
            current_q_info = torch.tensor([current_q_discrimination, current_q_difficulty, current_q_response, current_q_response_time]).to(device)
            trajectories[i][j] = current_q_info
            j += 1

            if current_q_response == 0:
                current_q = current_q.left
            else:
                current_q = current_q.right
    
    return trajectories


def eval_dt(model_paths: list, val_or_test: str):
    _, val_set, test_set = dataset_dt()
    assert val_or_test in ["val", "test"]
    data_set = val_set if val_or_test == "val" else test_set

    useful_tables = [data_set.groupby("item_id")["discrimination"].first(), 
                     data_set.groupby("item_id")["difficulty"].first(),
                     data_set.pivot_table(index="student_id", columns="item_id", values="response"),
                     data_set.pivot_table(index="student_id", columns="item_id", values="response_time"),
                     data_set["student_id"].unique()]

    abilities = data_set["final_score"].unique()

    N = len(abilities)

    # regression model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg_model = RegressionModel().to(device)
    reg_model.load_state_dict(torch.load("AL_model_weights/val_loss_best.pt", weights_only=True, map_location=device))
    reg_model.eval()

    # evaluate
    RMSE_all = dict()
    dt_model = None
    for path in model_paths:
        try:
            with open(path, "rb") as f:
                dt_model = pickle.load(f)
        except:
            continue
        
        print(f"Evaluating {path.split("/")[1].split(".")[0]} on {val_or_test} ...", end=" ")

        # get students' answers
        trajectories = generate_trajectories(dt_model, useful_tables, device)

        # ability estimate
        with torch.no_grad():
            RMSE = ( sum([ ( reg_model(trajectories[i].reshape(1, 244, 4)).item() - abilities[i] ) ** 2 for i in range(N)]) / N ) ** 0.5 
            RMSE_all[path] = RMSE
            print(f"{RMSE:.6f}")


        # batched data -- doesn't work (why?)

        # with torch.no_grad():
        #     estimated_abilities = reg_model(trajectories).cpu().numpy().reshape((N, ))
        #     RMSE = (np.sum((estimated_abilities - abilities) ** 2) / N) ** 0.5
        #     RMSE_all[ f"{config["model_name"]}_{config["mean_or_median"]}{"_T" if config["time_aware"] else ""}_D" ] = RMSE

    # display
    RMSE_all_sorted = dict(sorted(RMSE_all.items(), key=lambda x: x[1]))
    print("\n======== RMSE Summary ========")
    for path, RMSE in RMSE_all_sorted.items():
        print(f"{path.split("/")[1].split(".")[0]:<20}: {RMSE:.6f}")
    
    return RMSE_all_sorted



if __name__ == "__main__":
    model_paths = [
        os.path.join(
             "dt_models", 
             f"{model}{"_old" if choice == "all" else ""}_{mean_or_median}{"_T" if t_aware else ""}_D{".pkl"}"
        )     
        for model in ["c45", "id3", "cart"]
        for t_aware in [False, True]
        for mean_or_median in ["mean", "median"]
        for choice in ["all", "cond"]
    ]
    
    eval_dt(model_paths, "val")
