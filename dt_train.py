from dt_d_adaptive import DTDiscreteAdaptive
# from dt_d_static import DTDiscreteStatic
from dt_eval import eval_dt

import os
import time
import pickle


if __name__ == "__main__":
    dir_name = "dt_models"
    extension = ".pkl"
    
    models = ["id3", "c45", "cart"]
    num_classes_cand = [7, 8, 9, 10, 11, 12]

    time_aware = [False]
    data_repr = ["median"]
    time_choices = ["adaptive"]

    model_configs = [
        {"model_name": model, 
         "time_aware": t_aware, 
         "mean_or_median": mean_or_median,
         "time_choice": time_choice,
         "num_classes": num_classes,
         "path": os.path.join(
             dir_name, 
             f"{model}_{time_choice}_{mean_or_median}{"_T" if t_aware else ""}_D{num_classes}{extension}")
        }
        for model in models
        for t_aware in time_aware
        for mean_or_median in data_repr
        for time_choice in time_choices
        for num_classes in num_classes_cand
    ]

    data_set = "val"

    os.makedirs(dir_name, exist_ok=True)

    # evaluate and save all models
    RMSE_all = dict()
    for config in model_configs:
        if config["time_choice"] == "adaptive":
            dt_model = DTDiscreteAdaptive(model=config["model_name"], 
                                          time_aware=config["time_aware"], 
                                          mean_or_median=config["mean_or_median"],
                                          num_classes=config["num_classes"])
            
        # elif config["time_choice"] == "static":
        #     dt_model = DTDiscreteStatic(model=config["model_name"], 
        #                                   time_aware=config["time_aware"], 
        #                                   mean_or_median=config["mean_or_median"],
        #                                   num_classes=config["num_classes"])
        
        model_name = config["path"].split("/")[1].split(".")[0]

        # train
        start = time.time()
        print(f"Training {model_name} ... ", end="", flush=True)
        dt_model.train()
        end = time.time()
        print(f"{(end - start)/60:.2f} mins")

        # evaluate using ml model
        print(f"Evaluating {model_name} using ml model ... ", end="", flush=True)
        RMSE_ml = eval_dt(dt_model, data_set, "ml")
        print(f"{RMSE_ml:.6f}")

        # evaluate by itself
        print(f"Evaluating {model_name} by itself ... ", end="", flush=True)
        RMSE_self = eval_dt(dt_model, data_set, "self")
        print(f"{RMSE_self:.6f}")
        
        # save to dict
        RMSE_all[model_name] = [RMSE_ml, RMSE_self]

        with open(config["path"], "wb") as f:
            pickle.dump(dt_model, f)
        print(f"Saved to {config["path"]}\n")

    # summary
    print("=========== RMSE Summary ===========")
    print("ml evaluation:")
    RMSE_all_sorted = dict(sorted(RMSE_all.items(), key=lambda x: x[1][0]))
    for model_name, RMSE in RMSE_all_sorted.items():
        print(f"{model_name:<25}: {RMSE[0]:.6f}")
    print()

    print("self evaluation:")
    RMSE_all_sorted = dict(sorted(RMSE_all.items(), key=lambda x: x[1][1]))
    for model_name, RMSE in RMSE_all_sorted.items():
        print(f"{model_name:<25}: {RMSE[1]:.6f}")
    print()

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
