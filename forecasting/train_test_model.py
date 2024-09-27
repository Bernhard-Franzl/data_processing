import numpy as np
from _forecasting import list_checkpoints, run_n_tests, write_header_to_txt, evaluate_results, evaluate_results_lecture

  
cp_log_dir = "_forecasting/checkpoints/occrate"
#mode = "normal"
#filename = f"results_{mode}.txt"

##############################
# Run n tests

for data in ["train", "val"]:
    for mode in ["dayahead", "unlimited"]:
        filename = f"results_{mode}_occrate.txt"
        for run_id in [0]:
            
            raise NotImplementedError("Update forecast_iter in model.py")
            tuples_run_comb = list_checkpoints(cp_log_dir, run_id)

            list_combs, dict_losses, list_hyperparameters = run_n_tests(
                run_comb_tuples=tuples_run_comb,
                cp_log_dir=cp_log_dir, 
                mode=mode, 
                plot=False, 
                data=data,
                naive_baseline=False)   
                    
            list_combs = np.array(list_combs)

            write_header_to_txt(filename, run_id, data)

            evaluate_results(filename, 
                             list_combs, dict_losses, list_hyperparameters, top_k_params=5, baseline_losses=None)


##############################
# Test chosen combinations
#import torch
#for mode in ["dayahead", "unlimited"]:
#    filename = f"results_{mode}.txt"
#    for data in ["val"]:
        
#        list_combs, dict_losses, list_hyperparameters = run_n_tests(
#            run_comb_tuples=[(0,0)],
#            cp_log_dir=cp_log_dir, 
#            mode=mode, 
#            plot=True,  
#            data=data,
#            naive_baseline=False,
#        )   
        
#        # claculate mean losses
#        print(f"---------- {data} --------------")
#        for  i in range(len(list_combs)):
#            print(f"Combination: {list_combs[i]}")
#            for key in dict_losses:
#                mean_loss = torch.mean(torch.Tensor(dict_losses[key][i]))
#                print(f"Mean {key} loss: {mean_loss}")
#            print(f"------------------------")

###############################  Lecture Dataset ########################################

cp_log_dir = "_forecasting/checkpoints/lecture"
#cp_log_dir = "_forecasting/checkpoints/lecture"
#for data in ["train", "val"]:
#    for mode in ["time_sequential"]:
#        filename = f"_forecasting/results/results_lecture_{mode}.txt"
#        for run_id in [3]:
            
#            tuples_run_comb = list_checkpoints(cp_log_dir, run_id)

#            list_combs, dict_losses, list_hyperparameters, baseline_losses = run_n_tests(
#                run_comb_tuples=tuples_run_comb,
#                cp_log_dir=cp_log_dir, 
#                mode=mode, 
#                plot=False, 
#                data=data, 
#                naive_baseline=True)   
                    
#            list_combs = np.array(list_combs)

#            write_header_to_txt(filename, run_id, data)
#            evaluate_results_lecture(
#                filename=filename,
#                list_combs=list_combs, 
#                dict_losses=dict_losses, 
#                list_hyperparameters=list_hyperparameters, 
#                baseline_losses=baseline_losses, 
#                top_k_params=5)




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




