from _dfguru import DataFrameGuru as DFG
from _occupancy_forecasting import MasterTrainer
from _occupancy_forecasting import load_data
from _occupancy_forecasting import avoid_name_conflicts
from _evaluating import ParameterSearch

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import os
dfg = DFG()
torch.cuda.empty_cache()


############ Inputs ############
#args = parse_arguments()
#args = prompt_for_missing_arguments(args)0
#n_run = args.n_run
#n_param = args.n_param


for n_run in [1]:
    n_param = 0

    overwrite = True
    ################################

    param_dir = "_occupancy_forecasting/parameters/wrap_up_final"
    tb_log_dir = "_occupancy_forecasting/training_logs/wrap_up_final"
    cp_log_dir = "_occupancy_forecasting/checkpoints/wrap_up_final"
    path_to_data = "data/occupancy_forecasting"
    
    
    #raise NotImplementedError("This script is not up to date. Speedup in dataset generation is urgently needed.")
    # careful with overwriting features
    if overwrite:
        if os.path.exists(os.path.join(tb_log_dir, f"run_{n_run}")):
            os.system(f"rm -r {os.path.join(tb_log_dir, f"run_{n_run}")}")
        if os.path.exists(os.path.join(cp_log_dir, f"run_{n_run}")):
            os.system(f"rm -r {os.path.join(cp_log_dir, f"run_{n_run}")}")
            
    path_to_params = os.path.join(param_dir, f"run-{n_run}-{n_param}_params.json")
    
    ### load all feature combinations
    #import json 
    #with open(path_to_params, "r") as file:
    #    parameter_dict = json.load(file)

    #with open("all_combinations.json", "r") as f:
    #    features = json.load(f)
    #parameter_dict["features"] = features

    start_comb = avoid_name_conflicts(tb_log_dir, cp_log_dir, n_run)

    comb_iterator = ParameterSearch(path_to_json=path_to_params).grid_search_iterator(tqdm_bar=True)
    for n_comb, hyperparameters in enumerate(comb_iterator, start=start_comb):

        tb_path = os.path.join(tb_log_dir, f"run_{n_run}/comb_{n_comb}")
        cp_path = os.path.join(cp_log_dir, f"run_{n_run}/comb_{n_comb}")
        
        #### Control Randomness ####
        torch_rng = torch.Generator()
        torch_rng.manual_seed(42)
        
        train_dict, val_dict, test_dict = load_data(
            path_to_data, 
            hyperparameters["frequency"], 
            split_by=hyperparameters["split_by"],
            dfguru=dfg,
            with_examweek=hyperparameters["with_examweek"]
        )

        writer = SummaryWriter(
            log_dir=tb_path,
        )
        
        mt = MasterTrainer(
            hyperparameters=hyperparameters,
            summary_writer=writer,
            torch_rng=torch_rng,
            cp_path=cp_path,
            path_to_helpers=path_to_data
        )
        
        mt.save_hyperparameters(save_path=cp_path)
        
        train_loader, val_loader, test_loader, model, optimizer = mt.initialize_all(
            train_dict, 
            val_dict, 
            test_dict
        )
        
        # train model for n_updates
        mt.train_n_updates(train_loader, val_loader, 
                            model, optimizer, log_predictions=False)
        

        # Final Test on Validation and Training Set -> for logging purposes
        model, _,  _ =  mt.load_checkpoint(cp_path)
        mt.criterion = nn.L1Loss()
        mt.test_one_epoch(val_loader, model, log_info=True)
        val_loss_final = mt.stats_logger.val_loss.pop()
        mt.test_one_epoch(train_loader, model, log_info=True)
        train_loss_final = mt.stats_logger.val_loss.pop()
        
        # Write final losses to tensorboard
        mt.hyperparameters_to_writer(
            val_loss=np.mean(val_loss_final), 
            train_loss=np.mean(train_loss_final)
        )
            
        writer.close()
