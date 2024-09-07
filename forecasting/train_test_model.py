
from _forecasting import MasterTrainer, load_data_dicts
from torch.optim import Adam
import torch
import plotly.graph_objects as go
from torch.utils.tensorboard import SummaryWriter

from _dfguru import DataFrameGuru as DFG
# specify run and combination number

# Test run 0
# 0,7; 
n_run, n_comb = 4,0

# load model, optimizer, data sets and hyperparameters
torch_rng = torch.Generator()
torch_rng.manual_seed(42)
dfg = DFG()

writer = SummaryWriter(
    log_dir=f"_forecasting/testing_logs/run_{n_run}/comb_{n_comb}",
)
    
    
mt = MasterTrainer(
    optimizer_class=Adam,
    hyperparameters={"model_class":"simple_lstm",
                     "criterion":"MSE",},
    torch_rng=torch_rng,
    summary_writer=writer,
)

model, optimizer,  hyperparameters = mt.load_checkpoint(
    f"_forecasting/checkpoints/run_{n_run}/comb_{n_comb}"
    )
mt.set_hyperparameters(hyperparameters)

train_dict, val_dict, test_dict = load_data_dicts("data", hyperparameters["frequency"], dfguru=dfg)
train_set, val_set, test_set = mt.initialize_dataset(train_dict, val_dict, test_dict)
train_loader, val_loader, test_loader = mt.initialize_dataloader(train_set, val_set, test_set)

mt.test_one_epoch(train_loader, model, log_info=True)
mt.test_one_epoch(val_loader, model, log_info=True)


losses = mt.stats_logger.val_loss
predictions = mt.stats_logger.val_pred
targets = mt.stats_logger.val_target
inputs = mt.stats_logger.val_input
infos = mt.stats_logger.val_info


import numpy as np
import pandas as pd

for i, type in enumerate(["train", "val"]):

    pred_i = torch.cat(predictions[i], dim=0)
    target_i = torch.cat(targets[i], dim=0)
    
    y_times = []
    for x in infos[i]:
        #room_id, x_time, y_time, features, room_capa = x
        for room_id, x_time, y_time, _, room_capa in x:
            y_times.append(y_time.values[0])
    
    data = pd.DataFrame(columns=["time", "pred", "target"],
                        data={"time":y_times, "pred":pred_i.squeeze(), "target":target_i.squeeze()})

    plot_data = data.sort_values(by="time")

    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=plot_data["time"],
            y=plot_data["pred"],
            mode="lines+markers",
            name="Prediction"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=plot_data["time"],
            y=plot_data["target"],
            mode="lines+markers",
            name="Target"
        )
    )
    
    fig.show()

    #x_i = torch.cat([x for x, _ in inputs[i]], dim=0)
    #y_features_i = torch.cat([y for _, y in inputs[i]], dim=0)
    
    #losses_i = elementwise_mse(pred_i, target_i)
    
    #top_k = torch.topk(losses_i, 5, largest=True).indices
    #bot_k = torch.topk(losses_i, 5, largest=False).indices

    #mean_loss =  torch.mean(torch.Tensor(losses[i]))
        
    #info_flat = []
    #for x in infos[i]:
    #    info_flat.extend(x)
        
    
    ## Top 5
    #fig = go.Figure()        
    
    #for k in top_k:

    #    room_k, time_x_k, time_y_k, exogen_features_k, room_capa_k = info_flat[k]
    #    fig.add_trace(
    #        go.Scatter(
    #            x=time_y_k, 
    #            y=pred_i[k],
    #            mode="lines+markers+text",
    #            name=f"Prediction {k}"))


    #    fig.add_trace(
    #        go.Scatter(
    #            x=time_y_k, 
    #            y=target_i[k],
    #            mode="lines+markers+text",
    #            name=f"Target {k}"))

    #    fig.add_trace(
    #        go.Scatter(
    #            x=time_x_k, 
    #            y=x_i[k][:,0],
    #            mode="lines+markers+text",
    #            name=f"Input X {k}"))

    #    for j, exogen_feature in enumerate(exogen_features_k):
    #        fig.add_trace(
    #            go.Scatter(
    #                x=time_y_k, 
    #                y=y_features_i[k][:,j],
    #                mode="lines+markers+text",
    #                name=f"Y-feature:{exogen_feature} {k}"))

    ## add title
    #fig.update_layout(
    #    title_text=f"Top 5: {type} losses. Mean loss: {mean_loss}"
    #)

    #fig.show()
    
    ## Bot 5
    #fig = go.Figure()
    #for k in bot_k:
    #    room_k, time_x_k, time_y_k, exogen_features_k, room_capa_k = info_flat[k]
    #    fig.add_trace(
    #        go.Scatter(
    #            x=time_y_k, 
    #            y=pred_i[k],
    #            mode="lines",
    #            name=f"Prediction {k}"))

    #    fig.add_trace(
    #        go.Scatter(
    #            x=time_y_k, 
    #            y=target_i[k],
    #            mode="lines",
    #            name=f"Target {k}"))
        
    #    fig.add_trace(
    #        go.Scatter(
    #            x=time_x_k, 
    #            y=input_i[k][:, 0],
    #            mode="lines",
    #            name=f"Input {k}"))
    #fig.update_layout(
    #    title_text=f"Bot 5: {type} losses. Mean loss: {mean_loss}"
    #)
    #fig.show()
            




