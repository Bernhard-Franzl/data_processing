import numpy as np 

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from _forecasting.data import OccupancyDataset

class MasterTrainer:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_updates = 0
    val_interval = 250
    
    def __init__(self, model_class:nn.Module, optimizer:Optimizer, hyperparameters:dict, criterion) -> None:
        
        self.model_class = model_class
        self.criterion = criterion
        self.optimizer = optimizer
        self.hyperparameters = hyperparameters
        
    def reset_n_updates(self) -> None:
        self.n_updates = 0

    def update_hyperparameters(self, hyperparameters:dict) -> None:
        self.hyperparameters.update(hyperparameters)
        
    def custom_collate(self, x):
        info = [x[0] for x in x]
        X = torch.stack([x[1] for x in x])
        y = torch.stack([x[2] for x in x])
        return info, X, y
      
    ######## Initialization ########
    def initialize_model(self) -> nn.Module:
        model = self.model_class(**self.hyperparameters)
        return model.to(self.device)    
    
    def initialize_optimizer(self, model:nn.Module) -> Optimizer:
        
        if self.optimizer == torch.optim.Adam:
            optimizer = self.optimizer(model.parameters(), lr=self.hyperparameters["lr"])
        elif self.optimizer == torch.optim.SGD:
            optimizer = self.optimizer(model.parameters(), lr=self.hyperparameters["lr"], momentum=self.hyperparameters["momentum"])
        else:
            raise ValueError("Optimizer not supported.")
        
        return optimizer
    
    def initialize_dataset(self, train_dict:dict, val_dict:dict, test_dict:dict, frequency:str):
        
        train_set = OccupancyDataset(train_dict, frequency, self.hyperparameters["x_size"], self.hyperparameters["y_size"])
        val_set = OccupancyDataset(val_dict, frequency, self.hyperparameters["x_size"], self.hyperparameters["y_size"])
        test_set = OccupancyDataset(test_dict, frequency, self.hyperparameters["x_size"], self.hyperparameters["y_size"])
        
        return train_set, val_set, test_set
        
    def initialize_dataloader(self, train_set:Dataset, val_set:Dataset, test_set:Dataset):
        train_loader = DataLoader(train_set, batch_size=self.hyperparameters["batch_size"], shuffle=True, collate_fn=self.custom_collate)
        val_loader = DataLoader(val_set, batch_size=self.hyperparameters["batch_size"], shuffle=True, collate_fn=self.custom_collate)
        test_loader = DataLoader(test_set, batch_size=self.hyperparameters["batch_size"], shuffle=True, collate_fn=self.custom_collate)
        return train_loader, val_loader, test_loader
        
    ######## Training ########
    def train(self, dataloader:DataLoader, model:nn.Module, optimizer:Optimizer, val_dataloader:DataLoader=None):
        """
        trains a model for one epoch
        """
        
        train_loss = []
        val_loss = []
        
        for info, X, y in dataloader:
            
            optimizer.zero_grad()
        
            X = X.to(self.device)
            y = y.to(self.device)
            
            model_output = model(X)
            loss = self.criterion(model_output, y)
            
            loss.backward()
            optimizer.step()
            
            self.n_updates += 1
            
            # validate every 100 updates
            if (self.n_updates % self.val_interval == 0) and val_dataloader:
                val_losses = self.validate(val_dataloader, model)
                val_loss.append(np.mean(val_losses))
                
            train_loss.append(loss.cpu().detach())
            
        return train_loss, val_loss
    
    def validate(self, dataloader:DataLoader, model:nn.Module):
        """
        validates a model for one epoch
        """
        
        val_loss = []
        predictions = []
        with torch.no_grad():
            for info, X, y in dataloader:
                
                X = X.to(self.device)
                y = y.to(self.device)
                
                model_output = model(X)
                loss = self.criterion(model_output, y)
                
                val_loss.append(loss.cpu().detach())
                predictions.append(model_output.cpu().detach())
                
        return val_loss