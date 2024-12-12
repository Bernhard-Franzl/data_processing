import numpy as np
from _occupancy_forecasting import list_checkpoints, run_n_tests, write_header_to_txt, evaluate_results, evaluate_results_lecture
import os
import torch
from _occupancy_forecasting.testing import load_checkpoint, run_detailed_test_forward

from _dfguru import DataFrameGuru as DFG
from _occupancy_forecasting import MasterTrainer, LectureDataset
import json
        
        
cp_log_dir = "_forecasting/checkpoints/lecture"
dfg = DFG()

for run_id in [4]:
    for mode in ["time_sequential"]:
        filename = f"_forecasting/results/results_lecture_{mode}.txt"
                
                
        tuples_run_comb = sorted(list_checkpoints(cp_log_dir, run_id))

        # actually iterate over tuples, but we only have one combination so it doesnt matter
        
        all_stuff_list = []
        for i in range(3, 13):
    
            #tb_path = os.path.join(tb_log_dir, f"run_{run_id}/comb_{n_comb}_data_{i}")
            #cp_path = os.path.join(cp_log_dir, f"run_{run_id}/comb_{n_comb}_data_{i}")
            
            torch_rng = torch.Generator()
            torch_rng.manual_seed(42)
            
            # load data from checkpoint i
            datadf = dfg.load_dataframe(
                path_repo="data", 
                file_name=f"lecture_data_random_{i}"
            )
            checkpoint_path = os.path.join(cp_log_dir, f"run_{run_id}/comb_{0}_data_{i}")
            hyperparameters = json.load(open(os.path.join(checkpoint_path, "hyperparameters.json"), "r"))
            dataset = LectureDataset(datadf, 
                                     hyperparameters, 
                                     mode, 
                                     path_to_helpers=f"data/helpers_lecture_random_{i}.json", 
                                     validation=True)
            
            # load model from checkpoint i-1
            checkpoint_path = os.path.join(cp_log_dir, f"run_{run_id}/comb_{0}_data_{i-1}")
            hyperparameters = json.load(open(os.path.join(checkpoint_path, "hyperparameters.json"), "r"))
            model = load_checkpoint(checkpoint_path, False, hyperparameters)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            min_registered = dataset.min_registered
            max_registered = dataset.max_registered
            min_occrate = dataset.min_occrate
            max_occrate = dataset.max_occrate
            
            losses, predictions, infos, targets, inputs, target_features = run_detailed_test_forward(model, dataset, device)

            # denormalize
            print(min_occrate, max_occrate)
            pred_denorm = (np.array(predictions).squeeze() * (max_occrate - min_occrate)) + min_occrate
            target_denorm = (np.array(targets).squeeze()* (max_occrate - min_occrate)) + min_occrate
            
            lec_id_list = []
            y_starttime_list = []
            y_room_id_list = []
            for x in infos:
                lecture_id, _, y__starttime,_, _, y_room_id, _ = x
                lec_id_list.append(lecture_id)
                y_starttime_list.append(y__starttime)
                y_room_id_list.append(y_room_id)
                
            all_stuff = np.vstack((np.array(lec_id_list), np.array(y_starttime_list), np.array(y_room_id_list), pred_denorm, target_denorm)).T
            all_stuff_list.append(all_stuff)
        
        all_stuff = np.vstack(all_stuff_list)
        print(losses["MAE"][-1], predictions[-1], targets[-1])
        np.save(f"data/lecture_maxoccrate_estimates.npy", all_stuff)
        print(all_stuff[-1])

# Test chosen combinations
#import torch
#cp_log_dir = "_forecasting/checkpoints/lecture"
#for mode in ["time_onedateahead"]:
#    filename = f"results_lecture_{mode}.txt"
#    for data in ["train", "val"]:
        
#        list_combs, dict_losses, list_hyperparameters, baseline_losses = run_n_tests(
#            run_comb_tuples=[(2,2)],
#            cp_log_dir=cp_log_dir, 
#            mode=mode, 
#            plot=True,  
#            data=data,
#            naive_baseline=True,
#        )   
        
#        # claculate mean losses
#        print(f"---------- {data} --------------")
#        for  i in range(len(list_combs)):
#            print(f"Combination: {list_combs[i]}")
#            for key in dict_losses:
#                if key == "R2":
#                    continue
#                mean_loss = torch.mean(torch.Tensor(dict_losses[key][i]))
#                baseline_mean_loss = torch.mean(torch.Tensor(baseline_losses[key][i]))
#                print(f"Mean {key} loss: {mean_loss} | Baseline: {baseline_mean_loss}")
#            print(f"------------------------")




