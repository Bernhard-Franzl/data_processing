import numpy as np
from _occupancy_forecasting import OccupancyTestSuite
import torch
  

##############################
# Run n tests
cp_log_dir = "_occupancy_forecasting/checkpoints/occrate"

for mode in ["dayahead"]:
    for run_id in [6]:
   
        test_suite = OccupancyTestSuite(
            cp_log_dir=cp_log_dir,
            path_to_data="data/occupancy_forecasting",
            path_to_helpers="data/occupancy_forecasting",
            path_to_results=f"results_{mode}.txt",
            erase_results_file=False,
        )
        tuples_run_comb = sorted(test_suite.list_checkpoints(run_id))

        test_suite.evaluate_combinations(
            comb_tuples=tuples_run_comb,
            print_results=False,
            plot_results=True,
            dataset_mode=mode,

        ) 

        test_suite.writer.write_header(run_id)
        
        test_suite.analyse_results(
            hyperparameter_keys=[
                "split_by", 
                "dataset_mode", 
                "with_examweek", 
                "hidden_size", 
                "num_layers", 
                "x_horizon", 
                "y_horizon", 
                "course_encoding_dim", 
                "dropout", 
                "features"
            ]
        )


