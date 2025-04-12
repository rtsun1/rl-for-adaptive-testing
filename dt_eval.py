# look up table for evaluation
import time
import os
import pickle
import numpy as np
import pandas as pd

import torch
import torch.torch_version

from dataloader import dataset_rl as dataset_dt
from dt_d_adaptive import QuestionNode, DTDiscreteAdaptive
# from dt_d_static import QuestionNode, DTDiscreteStatic
from ability_level_model import RegressionModel

MAX_QUESTIONS = 15
MAX_TIME = 900

NUM_QUESTIONS = 244



def generate_trajectories(model, useful_tables, device):
    '''
    W.r.t. the decision tree, for each student, record the question information (dicrimination, difficulty, repsonse, time) for each question answered
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



def get_ability_estimates_itself(model, useful_tables):
    '''
    Estimate the ability by itself
    '''

    [_, _, response_table, time_table, stu_ids] = [useful_tables[i] for i in range(5)]

    N = len(response_table)
    ability_estimates = list()

    for i in range(N):
        time_remain = MAX_TIME
        num_qs = 0
        current_q = model.root_q

        while current_q is not None:
            current_q_id = current_q.item_id

            current_q_response = response_table.loc[stu_ids[i], current_q_id]
            current_q_response_time = time_table.loc[stu_ids[i], current_q_id]

            num_qs += 1
            time_remain -= current_q_response_time
            if time_remain < 0 or num_qs > MAX_QUESTIONS:
                ability_estimates.append(current_q.ability)
                break

            if current_q_response == 0:
                if current_q.left is None:
                    ability_estimates.append(current_q.ability)
                    break
                current_q = current_q.left
            else:
                if current_q.right is None:
                    ability_estimates.append(current_q.ability)
                    break
                current_q = current_q.right
    
    return np.array(ability_estimates)
    


def eval_dt(dt_model, val_or_test: str, eval_model: str):
    '''
    Compute the RMSE between the ability estimate and the true ability, where the ability estimation is either using the ml model or by itself
    '''

    _, val_set, test_set = dataset_dt()
    assert val_or_test in ["val", "test"]
    assert eval_model in ["self", "ml"]
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

    # ability estimate
    if eval_model == "ml":
        trajectories = generate_trajectories(dt_model, useful_tables, device)
        with torch.no_grad():
            # residuals = [ reg_model(trajectories[i].reshape(1, 244, 4)).item() - abilities[i] for i in range(N) ]
            # residuals_df = pd.DataFrame({"residual": residuals})

            RMSE = ( sum([ ( reg_model(trajectories[i].reshape(1, 244, 4)).item() - abilities[i] ) ** 2 for i in range(N)]) / N ) ** 0.5        # batched data doesn't work
            return RMSE #, residuals_df
    
    else:
        ability_estimates = get_ability_estimates_itself(dt_model, useful_tables)
        RMSE = (np.sum((ability_estimates - abilities) ** 2) / N) ** 0.5
        return RMSE



if __name__ == "__main__":
    dt_model_1, dt_model_2 = None, None
    with open("dt_models/cart_adaptive_median_D11.pkl", "rb") as f:
        dt_model_1 = pickle.load(f)
    
    with open("dt_models/id3_adaptive_median_T_D8.pkl", "rb") as f:
        dt_model_2 = pickle.load(f)
    
    rmse_1, residuals_df_1 = eval_dt(dt_model_1, "test", "ml")
    rmse_2, residuals_df_2 = eval_dt(dt_model_2, "test", "ml")

    print(f"{rmse_1:.6f}")
    print(f"{rmse_2:.6f}")

    # write residuals in csv
    os.makedirs("dt_results", exist_ok=True)
    residuals_df_1.to_csv("dt_results/dt_residuals_no_T.csv", index=False)
    residuals_df_2.to_csv("dt_results/dt_residuals_T.csv", index=False)

