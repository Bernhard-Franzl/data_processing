import numpy as np
from _occupancy_forecasting import OccupancyTestSuite
import torch
  

##############################
# Run n tests
#cp_log_dir = "_occupancy_forecasting/checkpoints/occrate"
#cp_log_dir = "_occupancy_forecasting/checkpoints/wrap_up_final"

mode = "normal"

for comb_id in [1]: # [1,2,3]
    cp_log_dir_run = f"/home/berni/transfer_from_server/wrap_up_combs_{comb_id}"
    for run_id in [0]:

        #if comb_id == 2:
        #    if run_id < 4:
        #        continue
        
        print(comb_id, run_id)
        
        test_suite = OccupancyTestSuite(
            cp_log_dir=cp_log_dir_run,
            path_to_data="data/occupancy_forecasting",
            path_to_helpers="data/occupancy_forecasting",
            path_to_results=f"results_server_{comb_id}_test.txt",
            erase_results_file=False,
        )
        tuples_run_comb = sorted(test_suite.list_checkpoints(run_id))

        test_suite.evaluate_combinations(
            comb_tuples=tuples_run_comb,
            print_results=False,
            plot_results=False,
            dataset_mode=mode,
        ) 

        test_suite.writer.write_header(run_id)
        
        test_suite.analyse_results(
            hyperparameter_keys=[
                #"split_by", 
                #"dataset_mode", 
                #"with_examweek", 
                #"hidden_size", 
                #"num_layers", 
                #"x_horizon", 
                #"y_horizon", 
                #"course_encoding_dim", 
                #"dropout", 
                "features"
            ]
        )


