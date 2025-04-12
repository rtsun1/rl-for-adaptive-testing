from dt import DecisionTree
from dt_eval import eval_dt

import os
import time
import pickle
import numpy as np
import pandas as pd


if __name__ == "__main__":
    model_dir_name = "dt_models"
    extension = ".pkl"
    
    models = ["cart"]
    num_classes_cand = [11]

    time_aware = [False]
    data_repr = ["median"]

    max_time_min_set = np.arange(1, 31, 3, dtype=int)
    max_qs_set = [np.inf]

    model_configs = [
        {"model_name": model, 
         "time_aware": t_aware, 
         "mean_or_median": mean_or_median,
         "num_classes": num_classes,
         "max_time_min": max_time_min,
         "max_qs": max_qs,
         "path": os.path.join(
             model_dir_name, 
             f"{model}_{num_classes}_{mean_or_median}{"_T" if t_aware else ""}_{max_time_min}{extension}")
        }
        for model in models
        for t_aware in time_aware
        for mean_or_median in data_repr
        for num_classes in num_classes_cand
        for max_time_min in max_time_min_set
        for max_qs in max_qs_set
    ]

    data_set = "test"
    results_dir_name = "dt_results"

    os.makedirs(model_dir_name, exist_ok=True)
    os.makedirs(results_dir_name, exist_ok=True)

    results_dict = dict()

    # save and eval
    for config in model_configs:
        dt_model = DecisionTree(model=config["model_name"], 
                                time_aware=config["time_aware"], 
                                mean_or_median=config["mean_or_median"],
                                num_classes=config["num_classes"],
                                max_time=config["max_time_min"]*60,
                                max_qs=config["max_qs"])
        
        model_name = config["path"].split("/")[1].split(".")[0]

        # train
        start = time.time()
        print(f"Training {model_name} ... ", end="", flush=True)
        dt_model.train()
        end = time.time()
        print(f"{(end - start)/60:.2f} mins")

        # save
        with open(config["path"], "wb") as f:
            pickle.dump(dt_model, f)
        print(f"Saved to {config["path"]}\n")

        # eval by ml
        print(f"Evaluating {model_name} using ml model ... ", end="", flush=True)
        RMSE_ml = eval_dt(dt_model, data_set, "ml")
        print(f"{RMSE_ml:.6f}")

        # eval by itself
        print(f"Evaluating {model_name} by itself ... ", end="", flush=True)
        RMSE_self = eval_dt(dt_model, data_set, "self")
        print(f"{RMSE_self:.6f}")

        # save results
        results_dict[config["max_time_min"]] = [RMSE_ml, RMSE_self]

        print()

    # save results into csv
    results_df = pd.DataFrame(results_dict).T
    results_df.reset_index(inplace=True)
    results_df.columns = ["time_limit", "ml", "self"]
    results_df.to_csv(os.path.join(results_dir_name, "dt_experiment.csv"), index=False)



'''
=========== RMSE Summary ===========
ml evaluation:
cart_adaptive_median_D11 : 0.450516
cart_adaptive_median_D8  : 0.450954
cart_adaptive_median_D9  : 0.451231
id3_adaptive_median_D12  : 0.451810
c45_adaptive_median_D8   : 0.451855
id3_adaptive_median_D9   : 0.453791
id3_adaptive_median_D8   : 0.453990
id3_adaptive_median_D7   : 0.454008
cart_adaptive_median_D12 : 0.454264
id3_adaptive_median_D10  : 0.454825
c45_adaptive_median_D7   : 0.457347
id3_adaptive_median_D11  : 0.457353
cart_adaptive_median_D10 : 0.458044
c45_adaptive_median_D10  : 0.458132
c45_adaptive_median_D12  : 0.458326
cart_adaptive_median_D7  : 0.458699
c45_adaptive_median_D11  : 0.459643
c45_adaptive_median_D9   : 0.460982

self evaluation:
id3_adaptive_median_D9   : 0.457605
id3_adaptive_median_D8   : 0.462124
cart_adaptive_median_D9  : 0.462414
cart_adaptive_median_D8  : 0.463119
cart_adaptive_median_D11 : 0.463483
id3_adaptive_median_D12  : 0.464518
id3_adaptive_median_D7   : 0.464552
id3_adaptive_median_D10  : 0.464757
c45_adaptive_median_D8   : 0.466018
id3_adaptive_median_D11  : 0.466794
c45_adaptive_median_D12  : 0.467981
c45_adaptive_median_D10  : 0.468961
c45_adaptive_median_D7   : 0.469568
cart_adaptive_median_D12 : 0.471447
cart_adaptive_median_D7  : 0.472962
c45_adaptive_median_D9   : 0.473686
c45_adaptive_median_D11  : 0.474399
cart_adaptive_median_D10 : 0.475571
'''
