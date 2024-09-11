
import torch
from torch.optim import Adam
import torchmetrics
from _forecasting import OccupancyDataset
import plotly.graph_objects as go
import numpy as np
import json
import os
from tqdm import tqdm

from _forecasting import SimpleOccDenseNet
from _forecasting import SimpleOccLSTM
from _forecasting import load_data_dicts


def load_checkpoint(checkpoint_path:str, load_optimizer:bool):
    
    # ignore warnings
    hyperparameters = json.load(open(os.path.join(checkpoint_path, "hyperparameters.json"), "r"))
    
    model_class = handle_model_class(hyperparameters["model_class"])
    model = model_class(hyperparameters)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt"), weights_only=True))
    
    if load_optimizer:
        optimizer = Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), weights_only=True))
        return model, hyperparameters, optimizer
    
    return model, hyperparameters

def prepare_model_and_data(checkpoint_path:str, dfg, device):
    
    model,  hyperparameters = load_checkpoint(
        checkpoint_path = checkpoint_path,
        load_optimizer = False
    )
    model = model.to(device)

    train_dict, val_dict, test_dict = load_data_dicts(
        "data", 
        hyperparameters["frequency"], 
        dfguru=dfg)
    
    room_ids = train_dict.keys()
    
    return model, hyperparameters, (train_dict, val_dict, test_dict), room_ids
    
def run_detailed_test(model, dataset:OccupancyDataset, device):
    
    model.eval()
    model = model.to(device)
    
    mae_f = torch.nn.L1Loss(reduction="mean")
    mse_f = torch.nn.MSELoss(reduction="mean")
    r2_f = torchmetrics.R2Score()      
    
    losses = {"MAE":[], "MSE":[], "RMSE":[], "R2":[]}
    
    predictions = []
    bar_format = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    for info, X, y_features, y in tqdm(dataset, total=len(dataset), bar_format=bar_format, leave=False):

        X = X.to(device)
        room_id = torch.IntTensor([info[0]]).to(device)
        y_features = y_features.to(device)
        
        with torch.no_grad():
            preds = model.forecast_iter(X, y_features, len(y), room_id)
            
            preds = preds.to("cpu")
            y_adjusted = y[:len(preds)]
            losses["MAE"].append(mae_f(preds, y_adjusted))
            losses["MSE"].append(mse_f(preds, y_adjusted))
            losses["RMSE"].append(torch.sqrt(mse_f(preds, y_adjusted)))
            losses["R2"].append(r2_f(preds, y_adjusted.squeeze()))
            
        predictions.append(preds)
        
    return losses, predictions

def plot_predictions(dataset:OccupancyDataset, predictions:list, room_ids:list, n_run:int, n_comb:int):
    
    y_times = dict([(room_id,[]) for room_id in room_ids])    
    preds = dict([(room_id,[]) for room_id in room_ids])
    targets = dict([(room_id,[]) for room_id in room_ids])
    
    for i,(info, X, y_features, y) in enumerate(dataset):

            room_id = info[0]
            y_time = info[2]
            
            pred = predictions[i].numpy()
            
            y_times[room_id].append(y_time.values)
            preds[room_id].append(pred)
            targets[room_id].append(y.squeeze().numpy())
            

    for room_id in room_ids:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(y_times[room_id]),
                y=np.concatenate(preds[room_id]),
                mode="lines+markers",
                name=f"Prediction Room {room_id}"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(y_times[room_id]),
                y=np.concatenate(targets[room_id]),
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
        
def handle_model_class(model_name:str):
        
        if model_name == "simple_densenet":
            return SimpleOccDenseNet
        
        elif model_name == "simple_lstm":
            return SimpleOccLSTM
        
        else:
            raise ValueError(f"Model {model_name} not recognized")
        



#def initialize_dataset(self, train_dict:dict, val_dict:dict, test_dict:dict, mode:str):
    
#    train_set = OccupancyDataset(train_dict, self.hyperparameters, mode)
#    val_set = OccupancyDataset(val_dict, self.hyperparameters, mode)
#    test_set = OccupancyDataset(test_dict, self.hyperparameters, mode)

    
#    return train_set, val_set, test_set