from _dfguru import DataFrameGuru as DFG
from _forecasting import MasterTrainer, load_data_lecture
from _forecasting import avoid_name_conflicts, parse_arguments, prompt_for_missing_arguments
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
#args = prompt_for_missing_arguments(args)
#n_run = args.n_run
#n_param = args.n_param

n_run = 4
n_param = 0
mode = "time_sequential"
overwrite = False

"occrate_registered_exam_test_tutorium_starttime_endtime_calendarweek_weekday_type_studyarea_ects_level"

# train best hyperparameter set on every week -> store results for training of other model
# [3, 10], [3, 0]
################################

param_dir = "_forecasting/parameters/lecture"
tb_log_dir = "_forecasting/training_logs/lecture"
cp_log_dir = "_forecasting/checkpoints/lecture"

if overwrite:
    if os.path.exists(os.path.join(tb_log_dir, f"run_{n_run}")):
        os.system(f"rm -r {os.path.join(tb_log_dir, f"run_{n_run}")}")
    if os.path.exists(os.path.join(cp_log_dir, f"run_{n_run}")):
        os.system(f"rm -r {os.path.join(cp_log_dir, f"run_{n_run}")}")

path_to_params = os.path.join(param_dir, f"run-{n_run}-{n_param}_params.json")

start_comb = avoid_name_conflicts(tb_log_dir, cp_log_dir, n_run)

comb_iterator = ParameterSearch(path_to_json=path_to_params).grid_search_iterator(tqdm_bar=True)
for n_comb, hyperparameters in enumerate(comb_iterator, start=start_comb):
    
    #### Control Randomness ####
    for i in range(2, 13):
    
        tb_path = os.path.join(tb_log_dir, f"run_{n_run}/comb_{n_comb}_data_{i}")
        cp_path = os.path.join(cp_log_dir, f"run_{n_run}/comb_{n_comb}_data_{i}")
        
        torch_rng = torch.Generator()
        torch_rng.manual_seed(42)
        
        datadf = dfg.load_dataframe(
            path_repo="data", 
            file_name=f"lecture_data_random_{i}"
        )
        train_indices = np.load(f"data/lecture_train_idx_random_{i}.npy")
        val_indices = np.load(f"data/lecture_val_idx_random_{i}.npy")
        test_indices = np.load(f"data/lecture_test_idx_random_{i}.npy")
        
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
        
        
        train_set = mt.initialize_lecture_dataset_deployment(datadf, mode)
        test_set = mt.initialize_lecture_dataset_deployment(datadf, mode)
        val_set = mt.initialize_lecture_dataset_deployment(datadf, mode)
        
        df_train = datadf.iloc[train_indices]
        df_val = datadf.iloc[val_indices]
        df_test = datadf.iloc[test_indices]
        
        train_samples = []
        val_samples = []
        test_samples = []
        for sample in train_set.samples:
            
            info, X, y_features, y = sample
            lecture_id,_, y_starttime,_, _, y_room_id, _ = info
            
            masked_train = df_train[(df_train["coursenumber"] == lecture_id) & (df_train["starttime"] == y_starttime) & (df_train["roomid"] == y_room_id)]
            masked_val = df_val[(df_val["coursenumber"] == lecture_id) & (df_val["starttime"] == y_starttime) & (df_val["roomid"] == y_room_id)]
            masked_test = df_test[(df_test["coursenumber"] == lecture_id) & (df_test["starttime"] == y_starttime) & (df_test["roomid"] == y_room_id)]
            
            # sanity check -> sample should be in exactly one dataset
            if len(masked_train) > 0 and len(masked_val) > 0:
                raise ValueError("Sample found in train and val dataset")
            
            if len(masked_train) > 0:
                train_samples.append(sample)
            elif len(masked_val) > 0:
                val_samples.append(sample)
            elif len(masked_test) > 0:
                test_samples.append(sample)
            else:
                raise ValueError("Sample not found in any dataset")

        train_set.samples = train_samples
        val_set.samples = val_samples
        test_set.samples = test_samples

        train_loader, val_loader, test_loader = mt.initialize_lecture_dataloader(train_set, val_set, test_set, mode)
        
        model = mt.initialize_model()
        optimizer = mt.initialize_optimizer(model)

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
            
        writer.close()
