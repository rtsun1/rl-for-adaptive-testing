from dt_adaptive import DTDiscreteAdaptive
from dt_static import DTDiscreteStatic
from dt_eval import eval_dt

import os
import time
import pickle


if __name__ == "__main__":
    dir_name = "dt_models"
    extension = ".pkl"
    
    models = ["id3", "c45", "cart"]
    time_aware = [False]
    data_repr = ["median"]
    num_classes_cand = [6, 7, 8, 9, 10, 11, 12]
    time_choices = ["adaptive", "static"]

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

    os.makedirs(dir_name, exist_ok=True)

    # find the best model
    best_RMSE = 10000
    dt_model = None

    RMSE_all = dict()

    for config in model_configs:
        if config["time_choice"] == "adaptive":
            dt_model = DTDiscreteAdaptive(model=config["model_name"], 
                                          time_aware=config["time_aware"], 
                                          mean_or_median=config["mean_or_median"],
                                          num_classes=config["num_classes"])
            
        elif config["time_choice"] == "static":
            dt_model = DTDiscreteStatic(model=config["model_name"], 
                                          time_aware=config["time_aware"], 
                                          mean_or_median=config["mean_or_median"],
                                          num_classes=config["num_classes"])
        
        model_name = config["path"].split("/")[1].split(".")[0]
        start = time.time()
        print(f"Training {model_name} ... ", end="", flush=True)
        dt_model.train()
        end = time.time()
        print(f"{(end - start)/60:.2f} mins")

        print(f"Evaluating {model_name} ... ", end="", flush=True)
        RMSE = eval_dt(dt_model, "val")
        RMSE_all[model_name] = RMSE
        print(f"{RMSE:.6f} (current best: {best_RMSE:.6f})")

        if RMSE < best_RMSE:
            best_RMSE = RMSE
            print(f"Best model so far! Saving to {config["path"]}")
            with open(config["path"], "wb") as f:
                pickle.dump(dt_model, f)

    # summary
    RMSE_all_sorted = dict(sorted(RMSE_all.items(), key=lambda x: x[1]))
    print("\n=========== RMSE Summary ===========")
    for model_name, RMSE in RMSE_all_sorted.items():
        print(f"{model_name:<25}: {RMSE:.6f}")
    print()
