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


_, val_set, test_set = dataset_dt()


discrimination_table = val_set.groupby("item_id")["discrimination"].first()
difficulty_table = val_set.groupby("item_id")["difficulty"].first()

response_table = val_set.pivot_table(index="student_id", columns="item_id", values="response")
time_table = val_set.pivot_table(index="student_id", columns="item_id", values="response_time")

N = len(response_table)


# prepare data for evaluation
def generate_trajectories(model, stu_ids, device):
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

    trajectories = torch.full(size=(N, NUM_QUESTIONS, 4), fill_value=-1).to(device)
    qs_sorted = val_set["item_id"].unique()
    qs_sorted.sort()

    for i in range(N):
        time_remain = MAX_TIME
        num_qs = 0
        current_q = model.root_q

        while current_q is not None:
            current_q_id = current_q.item_id

            current_q_discrimination = discrimination_table.loc[current_q_id]
            current_q_difficulty = difficulty_table.loc[current_q_id]
            current_q_response = response_table.loc[stu_ids[i], current_q_id]
            current_q_response_time = time_table.loc[stu_ids[i], current_q_id]

            num_qs += 1
            time_remain -= current_q_response_time
            if time_remain < 0 or num_qs >= MAX_QUESTIONS:
                break
            
            current_q_info = torch.tensor([current_q_discrimination, current_q_difficulty, current_q_response, current_q_response_time]).to(device)
            idx = np.where(qs_sorted == current_q_id)[0][0]
            trajectories[i][idx] = current_q_info

            if current_q_response == 0:
                current_q = current_q.left
            else:
                current_q = current_q.right
    
    return trajectories



if __name__ == "__main__":
    # validation
    stu_ids = val_set["student_id"].unique()
    abilities = val_set["final_score"].unique()

    # regression model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg_model = RegressionModel().to(device)
    reg_model.load_state_dict(torch.load("AL_model_weights/val_loss_best.pt", weights_only=True, map_location=device))
    reg_model.eval()

    dir_name = "dt_models"
    extension = ".pkl"

    # models = ["c45", "id3", "cart"]
    # time_aware = [False, True]
    # data_repr = ["mean", "median"]
    models = ["id3"]
    time_aware = [False, True]
    data_repr = ["mean", "median"]


    model_configs = [
        {"model_name": model, 
         "time_aware": t_aware, 
         "mean_or_median": mean_or_median,
         "file_name": os.path.join(
             dir_name, 
             f"{model}_{mean_or_median}{"_T" if t_aware else ""}_D{extension}")
        }
        for model in models
        for t_aware in time_aware
        for mean_or_median in data_repr
    ]

    RMSE_all = dict()
    
    for config in model_configs:
        dt_model = None
        with open(config["file_name"], "rb") as f:
            dt_model = pickle.load(f)

        # get students' answers
        trajectories = generate_trajectories(dt_model, stu_ids, device).to(device)     # torch.Size([4271, 244, 4])

        # ability estimate
        with torch.no_grad():
            RMSE = ( sum([ ( reg_model(trajectories[i].reshape(1, 244, 4)).item() - abilities[i] ) ** 2 for i in range(N)]) / N ) ** 0.5 
            RMSE_all[ f"{config["model_name"]}_{config["mean_or_median"]}{"_T" if config["time_aware"] else ""}_D" ] = RMSE


        # batched data -- doesn't work (why?)

        # with torch.no_grad():
        #     estimated_abilities = reg_model(trajectories).cpu().numpy().reshape((N, ))
        #     RMSE = (np.sum((estimated_abilities - abilities) ** 2) / N) ** 0.5
        #     RMSE_all[ f"{config["model_name"]}_{config["mean_or_median"]}{"_T" if config["time_aware"] else ""}_D" ] = RMSE

    # display
    for model, RMSE in RMSE_all.items():
        print(f"{model:<20}: {RMSE:.6f}")
