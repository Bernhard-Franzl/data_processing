from _dfguru import DataFrameGuru as DFG
from _occupany_forecasting import load_data_lecture
from _occupany_forecasting import avoid_name_conflicts, parse_arguments, prompt_for_missing_arguments
from _evaluating import ParameterSearch
from _occupany_forecasting import LectureDataset

import torch

import numpy as np
import os
dfg = DFG()

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# TODO:
# handle courses with same starting time
# Implement early+ stopping

def update_hyperparameters(hyperparameters, update_dict:dict) -> None:
    hyperparameters.update(update_dict)
    return hyperparameters
    
def initialize_lecture_dataset(train_df:dict, val_df:dict, test_df:dict, dataset_mode:str, hyperparameters):
    
    train_set = LectureDataset(train_df, hyperparameters, dataset_mode, validation=False)
    val_set = LectureDataset(val_df, hyperparameters, dataset_mode, validation=True)
    test_set = LectureDataset(test_df, hyperparameters, dataset_mode, validation=True)
    
    info, X, y_features, y = train_set[1]
    if len(X.shape) != 1:
        hyperparameters = update_hyperparameters(
            hyperparameters,
            {
            "x_size": int(X.shape[1]),
            "y_features_size": int(y_features.shape[1]), 
            "y_size": int(y.shape[1]),
            "immutable_size": int(info[6].shape[0])
            }
        )
    else:
        hyperparameters = update_hyperparameters(
            hyperparameters,
            {
            "x_size": int(X.shape[0]),
            "y_features_size": int(y_features.shape[0]), 
            "y_size": int(y.shape[0]),
            "immutable_size": int(info[6].shape[0])
            }
        )
    
    return train_set, val_set, test_set, hyperparameters


############ Inputs ############
#args = parse_arguments()
#args = prompt_for_missing_arguments(args)
#n_run = args.n_run
#n_param = args.n_param

n_run = 0
n_param = 0
mode = "time_onedateahead"
# overwrite = False

"occrate_registered_exam_test_tutorium_starttime_endtime_calendarweek_weekday_type_studyarea_ects_level"

################################

param_dir = "_forecasting/parameters/lecture"
tb_log_dir = "_forecasting/training_logs/lecture"
cp_log_dir = "_forecasting/checkpoints/lecture"

# if overwrite:
#     if os.path.exists(os.path.join(tb_log_dir, f"run_{n_run}")):
#         os.system(f"rm -r {os.path.join(tb_log_dir, f"run_{n_run}")}")
#     if os.path.exists(os.path.join(cp_log_dir, f"run_{n_run}")):
#         os.system(f"rm -r {os.path.join(cp_log_dir, f"run_{n_run}")}")

path_to_params = os.path.join(param_dir, f"run-sklearn-{n_run}-{n_param}_params.json")

start_comb = avoid_name_conflicts(tb_log_dir, cp_log_dir, "sklearn")
comb_iterator = ParameterSearch(path_to_json=path_to_params).grid_search_iterator(tqdm_bar=True)

for n_comb, hyperparameters in enumerate(comb_iterator, start=start_comb):
    
    tb_path = os.path.join(tb_log_dir, f"run_sklearn/comb_{n_comb}")
    cp_path = os.path.join(cp_log_dir, f"run_sklearn/comb_{n_comb}")
    
    import numpy as np
    # if overwrite delete old files at tb_path and cp_path
    #### Control Randomness ####
    rng = np.random.default_rng(seed=42)
    
    train_df, val_df, test_df = load_data_lecture("data", dfguru=dfg)
    
    trainset, val_set, test_set, hyperparameters = initialize_lecture_dataset(
        train_df, val_df, test_df, mode, hyperparameters)
    print("Length of trainset: ", len(trainset))    
    X_list = []
    y_features_list = []
    y_list = []
    course_numbers_list = []
    immutable_features_list = []
    for info, X, y_features, y in trainset:
        X_list.append(X.numpy())
        y_features_list.append(y_features.numpy())
        y_list.append(y.numpy())
        if "coursenumbers" in hyperparameters["features"]:
            raise
        immutable_features_list.append(info[6][:].numpy())

    X = np.stack(X_list)
    y_features = np.stack(y_features_list)
    immutable_features = np.stack(immutable_features_list)
    X = np.hstack([X, y_features, immutable_features])
    
    y = np.stack(y_list)
    
    if hyperparameters["model_class"]=="linearreg":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=True)
    
    elif hyperparameters["model_class"]=="ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(
            alpha=hyperparameters["alpha"],
            fit_intercept=hyperparameters["fit_intercept"], 
            copy_X=True, 
            solver='auto', 
            positive=False, 
            random_state=42)
        
    elif hyperparameters["model_class"]=="randomforest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=hyperparameters["n_estimators"],
            criterion=hyperparameters["criterion"], 
            max_depth=hyperparameters["max_depth"], 
            min_samples_split=hyperparameters["min_samples_split"], 
            min_samples_leaf=hyperparameters["min_samples_leaf"], 
            min_weight_fraction_leaf=0.0, 
            max_features=1.0, 
            max_leaf_nodes=None, 
            min_impurity_decrease=0.0, 
            bootstrap=hyperparameters["bootstrap"], 
            oob_score=False, 
            n_jobs=None, 
            random_state=42, 
            verbose=0, 
            warm_start=False, 
            ccp_alpha=0.0, 
            max_samples=None, 
            monotonic_cst=None)
    elif hyperparameters["model_class"]=="svr":
        from sklearn.svm import SVR
        model = SVR(
            kernel=hyperparameters["kernel"],
            epsilon=hyperparameters["epsilon"],
            C=hyperparameters["C"]
        )
    elif hyperparameters["model_class"]=="gaussian":
        from sklearn.gaussian_process import GaussianProcessRegressor
        model = GaussianProcessRegressor(
            kernel=None,
            alpha=hyperparameters["alpha"],
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=42)
    else:
        raise ValueError("Model not implemented")
    
    model = model.fit(X, y)
    X_hat = model.predict(X)
    print(hyperparameters)
    print( np.mean(np.abs(X_hat - y)))
    
    
    X_list = []
    y_features_list = []
    y_list = []
    course_numbers_list = []
    immutable_features_list = []
    for info, X, y_features, y in val_set:
        X_list.append(X.numpy())
        y_features_list.append(y_features.numpy())
        y_list.append(y.numpy())
        if "coursenumbers" in hyperparameters["features"]:
            raise
        immutable_features_list.append(info[6][:].numpy())

    X = np.stack(X_list)
    y_features = np.stack(y_features_list)
    immutable_features = np.stack(immutable_features_list)
    X = np.hstack([X, y_features, immutable_features])
    y = np.stack(y_list)
    
    X_hat = model.predict(X)
    print( np.mean(np.abs(X_hat - y)))
    
    
    # writer = SummaryWriter(
    #     log_dir=tb_path,
    # )
    
    # mt = MasterTrainer(
    #     hyperparameters=hyperparameters,
    #     summary_writer=writer,
    #     torch_rng=torch_rng,
    #     cp_path=cp_path,
    # )
    
    # #mt.save_hyperparameters(save_path=cp_path)
    
    # train_loader, val_loader, test_loader, model, optimizer = mt.intialize_all_lecture(
    #     train_df, val_df, test_df, mode)
    
    # # train model for n_updates
    # mt.train_n_updates(train_loader, val_loader, 
    #                     model, optimizer, log_predictions=False)
    
    # # Final Test on Validation and Training Set -> for logging purposes
    # mt.criterion = nn.L1Loss()
    # mt.test_one_epoch(val_loader, model, log_info=True)
    # val_loss_final = mt.stats_logger.val_loss.pop()
    # mt.test_one_epoch(train_loader, model, log_info=True)
    # train_loss_final = mt.stats_logger.val_loss.pop()
    # # Write final losses to tensorboard
    # mt.hyperparameters_to_writer(val_loss=np.mean(val_loss_final), train_loss=np.mean(train_loss_final))
        
    # writer.close()
    