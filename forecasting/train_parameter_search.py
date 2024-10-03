from _dfguru import DataFrameGuru as DFG
from _forecasting import MasterTrainer, load_data_dicts
from _forecasting import avoid_name_conflicts, parse_arguments, prompt_for_missing_arguments
from _evaluating import ParameterSearch

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import os
dfg = DFG()
torch.cuda.empty_cache()

# TODO:
# Implement early stopping


############ Inputs ############
#args = parse_arguments()
#args = prompt_for_missing_arguments(args)
#n_run = args.n_run
#n_param = args.n_param

n_run = 2
n_param = 0

mode = "normal"
overwrite = False
################################


param_dir = "_forecasting/parameters/occrate"
tb_log_dir = "_forecasting/training_logs/occrate"
cp_log_dir = "_forecasting/checkpoints/occrate"

if overwrite:
    if os.path.exists(os.path.join(tb_log_dir, f"run_{n_run}")):
        os.system(f"rm -r {os.path.join(tb_log_dir, f"run_{n_run}")}")
    if os.path.exists(os.path.join(cp_log_dir, f"run_{n_run}")):
        os.system(f"rm -r {os.path.join(cp_log_dir, f"run_{n_run}")}")
        
path_to_params = os.path.join(param_dir, f"run-{n_run}-{n_param}_params.json")

start_comb = avoid_name_conflicts(tb_log_dir, cp_log_dir, n_run)
comb_iterator = ParameterSearch(path_to_json=path_to_params).grid_search_iterator(tqdm_bar=True)

for n_comb, hyperparameters in enumerate(comb_iterator, start=start_comb):
    
    tb_path = os.path.join(tb_log_dir, f"run_{n_run}/comb_{n_comb}")
    cp_path = os.path.join(cp_log_dir, f"run_{n_run}/comb_{n_comb}")
    
    #### Control Randomness ####
    torch_rng = torch.Generator()
    torch_rng.manual_seed(42)
    
    train_dict, val_dict, test_dict = load_data_dicts("data", hyperparameters["frequency"], dfguru=dfg)
    
    writer = SummaryWriter(
        log_dir=tb_path,
    )
    
    mt = MasterTrainer(
        hyperparameters=hyperparameters,
        summary_writer=writer,
        torch_rng=torch_rng,
        cp_path=cp_path,
    )
    
    mt.save_hyperparameters(save_path=cp_path)
    
    train_loader, val_loader, test_loader, model, optimizer = mt.initialize_all(train_dict, val_dict, test_dict, mode)
    
    # train model for n_updates
    mt.train_n_updates(train_loader, val_loader, 
                        model, optimizer, log_predictions=False)
    
    # Final Test on Validation and Training Set -> for logging purposes
    mt.criterion = nn.L1Loss()
    mt.test_one_epoch(val_loader, model, log_info=True)
    val_loss_final = mt.stats_logger.val_loss.pop()
    mt.test_one_epoch(train_loader, model, log_info=True)
    train_loss_final = mt.stats_logger.val_loss.pop()
    # Write final losses to tensorboard
    mt.hyperparameters_to_writer(val_loss=np.mean(val_loss_final), train_loss=np.mean(train_loss_final))

    # save model, optimizer and hyperparameterscd
    mt.save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=cp_path
    )
        
    writer.close()
