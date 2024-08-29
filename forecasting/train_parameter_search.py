from _dfguru import DataFrameGuru as DFG
from _forecasting import OccFeatureEngineer
from _forecasting import OccupancyDenseNet
from _forecasting import MasterTrainer
from _forecasting import prepare_data

from torch import nn
from torch.optim import Adam

import numpy as np
from plotly import graph_objects as go

dfg = DFG()

############## Load Data & Derive Additional Features ################
n_run = 1
n_comb = 1
hyperparameters = {
        "lr": 0.001,
        "batch_size": 16,
        "hidden_size": 32,
        "x_size": 24,
        "y_size": 24,
        "features": "exam_lecture",
        "frequency": "1h",
        "n_epochs": 50
        
}

#################### Include Parameter Search ####################

train_dict, val_dict, test_dict = prepare_data(
    path_to_data_dir="data",
    hyperparameters=hyperparameters,
    dfguru = dfg
)

mt = MasterTrainer(
    model_class = OccupancyDenseNet,
    optimizer = Adam,
    hyperparameters = hyperparameters,
    criterion= nn.MSELoss()
)


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(
    log_dir=f"_forecasting/training_logs/run_{n_run}/comb_{n_comb}",
    filename_suffix=f"_run_{n_run}"
)

train_loader, val_loader, test_loader, model, optimizer = mt.initialize_all(train_dict, val_dict, test_dict)

mean_train_losses = []
val_losses = []
train_losses = []

for n_epoch in range(hyperparameters["n_epochs"]):

    train_loss, val_loss = mt.train(train_loader, model, optimizer, val_loader)
        
    mean_train_losses.append(np.mean(train_loss))
    
    val_losses.extend(val_loss)
    train_losses.extend(train_loss)
    
    writer.add_scalar("Loss/train", np.mean(train_loss), n_epoch)

for i, val_loss in enumerate(val_losses):
    writer.add_scalar("Loss/val", val_loss, i*mt.val_interval)

# validate on test set
import torch
writer.add_hparams(
    hparam_dict=hyperparameters, 
    metric_dict={'hparam/loss': torch.Tensor([1.2])}
)

#tensorboard --logdir=/home/berni/github_repos/data_processing/forecasting/_forecasting/training_logs

writer.close()