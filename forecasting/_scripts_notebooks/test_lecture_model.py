import numpy as np
from _lecture_forecasting import LectureTestSuite
import torch
  

###############################  Lecture Dataset ########################################

cp_log_dir = "_lecture_forecasting/checkpoints/lecture"
testsuite = LectureTestSuite(
    cp_log_dir=cp_log_dir,
    path_to_data="data/lecture_forecasting",
    path_to_helpers="data/lecture_forecasting",
    path_to_results="results.txt",
    erase_results_file=False,
)
#print("Warning: Batch size is set to 1 for testing purposes")


#for run_id in [3, 4, 5, 6]:
#    tuples_run_comb = testsuite.list_checkpoints(run_id)

#    combinations, hyperparameters, loss_dicts, baseline_loss_dicts, dataset_masks = testsuite.evaluate_combinations(
#        comb_tuples=tuples_run_comb,
#        plot_results=False,
#        )   
         
#    testsuite.writer.write_header(run_id)

#    testsuite.analyse_results(
#        combinations, hyperparameters, loss_dicts, baseline_loss_dicts, dataset_masks
#    )



#Plot a few combinations

combs_to_plot = [
     [3, 99], [3, 27], [3, 30], [3, 73], [3, 49],
     [4, 114], [4, 135], [4, 124], [4, 123], [4, 136],
     [5, 100], [5, 137], [5, 128], [5, 126], [5, 125],
     [6, 98], [6, 117], [6, 126], [6, 26], [6, 95], [6, 1],
     ]
combinations, hyperparameters, loss_dicts, baseline_loss_dicts, dataset_masks = testsuite.evaluate_combinations(
    comb_tuples=combs_to_plot,
    plot_results=True,
)   


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




