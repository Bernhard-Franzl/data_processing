from _dfguru import DataFrameGuru as DFG
from _forecasting import MasterTrainer, load_data_dicts

from _evaluating import ParameterSearch

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import numpy as np
dfg = DFG()

# TODO:
# Add more features, such as: registered students,other course features, -1 week
# Implement early stopping

############ Test with small run first ############
# 6 is a testrun
#path_to_json = "_forecasting/parameters/run-0-0_params.json"
path_to_json = "_forecasting/parameters/run-1-0_params.json"

splitted = path_to_json.split("/")[-1].split("_")[0].split("-")
n_run = int(splitted[-2])
start_comb = int(splitted[-1])
comb_iterator = ParameterSearch(path_to_json=path_to_json).grid_search_iterator(tqdm_bar=True)

for n_comb, hyperparameters in enumerate(comb_iterator, start=start_comb):
    
    
    #### Control Randomness ####
    torch_rng = torch.Generator()
    torch_rng.manual_seed(42)
    
    train_dict, val_dict, test_dict = load_data_dicts("data", hyperparameters["frequency"], dfguru=dfg)
    
    writer = SummaryWriter(
        log_dir=f"_forecasting/training_logs/run_{n_run}/comb_{n_comb}",
    )
    
    mt = MasterTrainer(
        optimizer_class=Adam,
        hyperparameters=hyperparameters,
        summary_writer=writer,
        torch_rng=torch_rng
    )
    
    train_loader, val_loader, test_loader, model, optimizer = mt.initialize_all(train_dict, val_dict, test_dict)
    
    #raise
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
        save_path=f"_forecasting/checkpoints/run_{n_run}/comb_{n_comb}"
    )
        
    writer.close()
