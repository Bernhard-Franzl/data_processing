from _dfguru import DataFrameGuru as DFG
from _lecture_forecasting import MasterTrainerLecture, load_data
from _lecture_forecasting import avoid_name_conflicts, parse_arguments, prompt_for_missing_arguments
from _evaluating import ParameterSearch

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
dfg = DFG()

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# TODO:
# handle courses with same starting time
# Implement early+ stopping



############ Inputs ############
#args = parse_arguments()
#args = prompt_for_missing_arguments(args   )
#n_run = args.n_run
#n_param = args.n_param

n_run = 0
n_param = 0
mode = "time_sequential"
overwrite = True

"occrate_registered_exam_test_tutorium_starttime_endtime_calendarweek_weekday_type_studyarea_ects_level"
################################

param_dir = "_lecture_forecasting/parameters/lecture"
tb_log_dir = "_lecture_forecasting/training_logs/lecture"
cp_log_dir = "_lecture_forecasting/checkpoints/lecture"

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
    
    # if overwrite delete old files at tb_path and cp_path
    #### Control Randomness ####
    torch_rng = torch.Generator()
    torch_rng.manual_seed(42)
    
    train_df, val_df, test_df = load_data("data/lecture_forecasting", dfguru=dfg, split_by="time_10")
    writer = SummaryWriter(
        log_dir=tb_path,
    )
    
    mt = MasterTrainerLecture(
        hyperparameters=hyperparameters,
        summary_writer=writer,
        torch_rng=torch_rng,
        cp_path=cp_path,
    )
    
    mt.save_hyperparameters(save_path=cp_path)
    
    train_loader, val_loader, test_loader, model, optimizer = mt.intialize_all(
        train_df, val_df, test_df, mode, "data/helpers_lecture_time_10.json")
    

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
