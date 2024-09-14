import torch
import torchmetrics
from torch.optim import Adam

import os
import json
import time

import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm

from _forecasting.model import SimpleOccDenseNet, SimpleOccLSTM
from _forecasting.data import OccupancyDataset
from _forecasting.data import load_data_dicts

from _dfguru import DataFrameGuru as DFG


############### Load checkpoints ################
def list_checkpoints(path_to_dir, run_id):
    
    path_to_run = os.path.join(path_to_dir, f"run_{run_id}")
    
    if os.path.exists(path_to_run):
        comb_ids = list(map(lambda x: int(x.split("_")[-1]), os.listdir(path_to_run)))
        run_comb_tuples = list(zip([run_id]*len(comb_ids), comb_ids))
        del comb_ids
        return run_comb_tuples
    
    else:
        raise ValueError(f"Checkpoints of run {run_id} do not exist")    
  
def handle_model_class(model_name:str):
        
        if model_name == "simple_densenet":
            return SimpleOccDenseNet
        
        elif model_name == "simple_lstm":
            return SimpleOccLSTM
        
        else:
            raise ValueError(f"Model {model_name} not recognized")
        
def load_checkpoint(checkpoint_path:str, load_optimizer:bool, hyperparameters:dict):
    
    # ignore warnings
    
    model_class = handle_model_class(hyperparameters["model_class"])
    model = model_class(hyperparameters)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt"), weights_only=True))
    
    if load_optimizer:
        optimizer = Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), weights_only=True))
        return model, hyperparameters, optimizer
    
    return model

def prepare_model_and_data(checkpoint_path:str, dfg, device, mode:str, data:str):
    
    hyperparameters = json.load(open(os.path.join(checkpoint_path, "hyperparameters.json"), "r"))

    train_dict, val_dict, test_dict = load_data_dicts(
        "data", 
        hyperparameters["frequency"], 
        dfguru=dfg)
    
    if data == "train":
        data_dict = train_dict
    elif data == "val":
        data_dict = val_dict
    elif data == "test":
        data_dict = test_dict
    else:
        raise ValueError(f"Data {data} not recognized")
            
    dataset = OccupancyDataset(data_dict, hyperparameters, mode)
    room_ids = data_dict.keys()
    
    model = load_checkpoint(
        checkpoint_path = checkpoint_path,
        load_optimizer = False,
        hyperparameters = hyperparameters,
    )
    model = model.to(device)
    
    return model, hyperparameters, dataset, room_ids
    
def run_detailed_test(model, dataset:OccupancyDataset, device):
    
    model.eval()
    model = model.to(device)
    
    mae_f = torch.nn.L1Loss(reduction="mean")
    mse_f = torch.nn.MSELoss(reduction="mean")
    r2_f = torchmetrics.R2Score()      
    
    losses = {"MAE":[], "MSE":[], "RMSE":[], "R2":[]}
    
    predictions = []
    infos = []
    inputs = []
    targets = []
    target_features = []
    
    bar_format = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    for info, X, y_features, y in tqdm(dataset, total=len(dataset), bar_format=bar_format, leave=False):

        X = X.to(device)
        room_id = torch.IntTensor([info[0]]).to(device)
        y_features = y_features.to(device)
        
        with torch.no_grad():
            
            preds = model.forecast_iter(X, y_features, len(y), room_id)
            
            if len(preds) == 0:
                continue
            
            preds = preds.to("cpu")
            y_adjusted = y[:len(preds)]
            
            
            info = (info[0], info[1], info[2][:len(preds)], info[3], info[4])
            
            #print(len(y_adjusted), info[2].shape, "pred:", len(preds))
            #if preds.shape != y_adjusted.shape:
            #    y_adjusted = y_adjusted.unsqueeze(-1)
                
            losses["MAE"].append(mae_f(preds, y_adjusted))
            losses["MSE"].append(mse_f(preds, y_adjusted))
            losses["RMSE"].append(torch.sqrt(mse_f(preds, y_adjusted)))
            if len(preds) == 1:
                losses["R2"].append(None)
            else:
                losses["R2"].append(r2_f(preds, y_adjusted))
            
        predictions.append(preds)
        infos.append(info)
        inputs.append(X)
        targets.append(y_adjusted)
        target_features.append(y_features)
        
    return losses, predictions, infos, targets, inputs, target_features

def run_n_tests(run_comb_tuples, cp_log_dir, mode, plot, data):
    
    dfg = DFG()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    list_combs = []
    dict_losses = {"MAE":[], "MSE":[], "RMSE":[], "R2":[]}
    list_hyperparameters = []

    bar_format = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    for n_run, n_comb in tqdm(run_comb_tuples, total=len(run_comb_tuples), bar_format=bar_format, leave=False):

        checkpoint_path = os.path.join(cp_log_dir, f"run_{n_run}", f"comb_{n_comb}")

        model, hyperparameters, dataset, room_ids = prepare_model_and_data(
            checkpoint_path=checkpoint_path, 
            dfg=dfg, 
            device=device,
            mode=mode,
            data=data)
        
        # print size of model
        
        #print(f"{hyperparameters["model_class"]} size: {sum(p.numel() for p in model.parameters())}")
        #if hyperparameters["model_class"] == "simple_lstm":
        #    # size of lstm layer
        #    print(f"Size of LSTM layer: {sum(p.numel() for p in model.lstm.parameters())}")

        list_hyperparameters.append(hyperparameters)

        # run detailed test
        losses, predictions, infos, targets, _, _ = run_detailed_test(model, dataset, device)
        
        for key in dict_losses:
            dict_losses[key].append(losses[key])
        list_combs.append((n_run, n_comb))
        
        if plot:
            plot_predictions(infos, predictions, targets, room_ids, n_run, n_comb)
            
    return list_combs, dict_losses, list_hyperparameters

def plot_predictions(infos:list, predictions:list, targets:list, room_ids:list, n_run:int, n_comb:int):
    
    dict_y_times = dict([(room_id,[]) for room_id in room_ids])    
    dict_preds = dict([(room_id,[]) for room_id in room_ids])
    dict_targets = dict([(room_id,[]) for room_id in room_ids])
    
    
    for i, pred in enumerate(predictions):

            room_id = infos[i][0]
            y_time = infos[i][2]
            
            pred = pred.numpy()
            
            dict_y_times[room_id].append(y_time.values)

            if pred.shape[-1]==1:
                pred = pred.squeeze(-1)
            
            dict_preds[room_id].append(pred)
            dict_targets[room_id].append(targets[i].squeeze().numpy())
            
    for room_id in room_ids:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(dict_y_times[room_id]),
                y=np.concatenate(dict_preds[room_id]),
                mode="lines+markers",
                name=f"Prediction Room {room_id}"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(dict_y_times[room_id]),
                y=np.concatenate(dict_targets[room_id]),
                mode="lines+markers",
                name=f"Target Room {room_id}"
            )
        )
        
        fig.update_layout(
            title=f"Run {n_run} - Combination {n_comb} - Room {room_id}",
            xaxis_title="Time",
            yaxis_title="Occupancy"
        )
        
        fig.show()
        
        
        
############## Write to txt file ####################
def write_header_to_txt(file_name, run_id, data):
    
    with open(file_name, "a") as file:
        file.write(f"#################\n")
        file.write(f"Data: {data}\n")
        file.write(f"Run: {run_id}\n")
        file.write(f"Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}\n")
    
def write_loss_to_txt(file_name, combinations, losses, loss_f):
    
    with open(file_name, "a") as file:
        file.write(f"Loss function: {loss_f}\n")
        file.write(f"Combinations: {combinations.tolist()}\n")
        file.write(f"Losses: {losses.tolist()}\n")
  
def write_new_line(file_name):
    with open(file_name, "a") as file:
        file.write("\n")

def write_hyperparameters_to_txt(file_name, hyperparameters):
    
    listy = [(key, hyperparameters[key])  for  key in sorted(hyperparameters.keys())]
    with open(file_name, "a") as file:
        file.write(f"Hyperparameters: {listy}\n")

def erase_file(file_name):
    with open(file_name, "w") as file:
        file.write("")      



############## Evaluate results ####################
def evaluate_results(filename, list_combs, dict_losses, list_hyperparameters, top_k_params):

    for key, value in dict_losses.items():
        
        mean_losses = np.array([torch.mean(torch.Tensor(x)) for x in value])
        # sort by mean loss, descending if R2 -> we sort best to worst
        if key == "R2":
            indices = np.argsort(mean_losses)[::-1]
        else:
            indices = np.argsort(mean_losses)

            
        write_loss_to_txt(filename, list_combs[indices], mean_losses[indices], key)
        
        list_hyperparameters_k = [list_hyperparameters[i] for i in indices[:top_k_params]]
        all_keys = all_keys = set().union(*list_hyperparameters_k)
        param_results = {}
        for key in all_keys:
            vc = np.unique([params_dict[key] for params_dict in list_hyperparameters_k], return_counts=True, axis=0)
            # make vc readable
            vc = list(zip(vc[0], vc[1]))
            param_results[key] = vc
          
        write_hyperparameters_to_txt(filename, param_results)  
        
    write_new_line(filename)
  
def get_k_smallest_largest(k:int, losses:dict):
    
    smallest_k = torch.topk(torch.Tensor(losses), k, largest=False).indices
    largest_k = torch.topk(torch.Tensor(losses), k, largest=True).indices
    
    return smallest_k, largest_k