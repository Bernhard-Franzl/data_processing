
import torch
import torchmetrics
from _forecasting import OccupancyDataset
import plotly.graph_objects as go
import numpy as np

def detailed_test(model, dataset:OccupancyDataset):
    
    model.eval()
    
    mae_f = torch.nn.L1Loss(reduction="mean")
    mse_f = torch.nn.MSELoss(reduction="mean")
    r2_f = torchmetrics.R2Score()      
    
    losses = {"MAE":[], "MSE":[], "RMSE":[], "R2":[]}
    
    predictions = []
    
    for i,(info, X, y_features, y) in enumerate(dataset):

        room_id = torch.IntTensor([info[0]])
        
        with torch.no_grad():
            preds = model.forecast_iter(X, y_features, len(y), room_id)
            
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