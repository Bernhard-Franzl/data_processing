import numpy as np 
import json
import os
import warnings
import tqdm
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from _occupancy_forecasting.data import OccupancyDataset
from _occupancy_forecasting.model import SimpleOccDenseNet, MultiHeadOccDenseNet, SimpleOccLSTM, EncDecOccLSTM, EncDecOccLSTMExperimental

class StatsLogger:
    
    def __init__(self) -> None:
        
        self.mean_train_loss = []
        self.train_loss = []
        self.train_loss_buffer = []
        
        self.val_loss= []
        self.val_pred = []
        self.val_input = []
        self.val_target= []
        self.val_info = []
        
        
    def reset_logger(self) -> None:
        self.mean_train_loss = []
        self.train_loss = []
        self.train_loss_buffer = []
        self.train_loss_mini_buffer = []
        
        self.val_loss= []
        self.val_pred = []
        self.val_input = []
        self.val_target= []
        self.val_info = []
        
    def free_memory(self) -> None:
        self.val_pred = []
        self.val_target= []
        self.val_input = []
        
    def append_train_loss(self, loss:float) -> None:
        self.train_loss.append(loss)
        self.train_loss_buffer.append(loss)
        
    def reset_train_loss_buffer(self) -> None:
        self.train_loss_buffer = []
    
    def append_mean_train_loss(self) -> None:
        self.mean_train_loss.append(np.mean(self.train_loss_buffer))
        self.reset_train_loss_buffer()
        
    def append_val_stats(self, loss:list, pred:list, target:list, input:list, info=None) -> None:
        self.val_loss.append(loss)
        self.val_pred.append(pred)
        self.val_input.append(input)
        self.val_target.append(target)
        if info:
            self.val_info.append(info)
           
class MasterTrainer:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_updates = 0
    test_interval = 250
    n_test = 0
    train_log_inteval = 50
    room_capacities = {0:164, 1:152}
    best_model = None
    best_loss = 1000
    
    def __init__(self,  hyperparameters:dict, cp_path:str, path_to_helpers:str, summary_writer=None, torch_rng=None) -> None:
        
        self.hyperparameters = hyperparameters
        self.model_class = self.handle_model_class(hyperparameters["model_class"])
        self.criterion = self.handle_criterion(hyperparameters["criterion"])
        self.optimizer_class = self.handle_optimizer(hyperparameters["optimizer_class"])
        self.path_to_helpers = path_to_helpers
        self.stats_logger = StatsLogger()
        self.cp_path = cp_path
        
        
        if summary_writer:
            self.summary_writer = summary_writer
            
        if torch_rng:
            self.torch_rng = torch_rng

    def reset_n_updates(self) -> None:
        self.n_updates = 0

    def update_hyperparameters(self, hyperparameters:dict) -> None:
        self.hyperparameters.update(hyperparameters)
    
    def set_hyperparameters(self, hyperparameters:dict) -> None:
        self.hyperparameters = hyperparameters
        
    def custom_collate(self, x):
        info = [x_i[0] for x_i in x]
        X = torch.stack([x_i[1] for x_i in x])
        y_features = torch.stack([x_i[2] for x_i in x])
        y = torch.stack([x_i[3] for x_i in x])
        
        if not (x[0][0][5][0] == None):
            X_course = torch.stack([x_i[0][5][0].to(torch.int32) for x_i in x])
            y_course = torch.stack([x_i[0][5][1].to(torch.int32)  for x_i in x])
            courses = torch.cat([X_course, y_course], dim=1)
            
            del X_course
            del y_course
        
            return info, X, y_features, y, courses
        else:
            return info, X,  y_features, y, torch.Tensor([0])
    
    def MBE(self, y_pred, y_true):
        return torch.mean(y_true - y_pred)
    
    def LogCosh(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred-y_true)))
    
    def handle_criterion(self, criterion:str):

        if criterion == "SSE":
            return nn.MSELoss(reduction="sum")
        elif criterion == "MSE":
            return nn.MSELoss()
        elif criterion == "MAE":
            return nn.L1Loss()
        elif criterion == "LogCosh":
            return self.LogCosh
        elif criterion == "MBE":
            return self.MBE
        elif criterion == "SAE":
            return nn.L1Loss(reduction="sum")
        elif criterion == "CE":
            return nn.CrossEntropyLoss()
        elif criterion == "BCE":
            return nn.BCELoss()
        else:
            raise ValueError("Criterion not supported.")
    
    def handle_optimizer(self, optimizer_class:str):
        if optimizer_class == "Adam":
            return torch.optim.Adam
        
        elif optimizer_class == "SGD":
            return torch.optim.SGD
        
        else:
            raise ValueError("Optimizer not supported.")
        
    def handle_model_class(self, model_class:str):
        
        if model_class == "simple_lstm":
            return SimpleOccLSTM

        elif model_class == "simple_densenet":
            return SimpleOccDenseNet
        
        elif model_class == "multihead_densenet":
            return MultiHeadOccDenseNet

        elif model_class == "ed_lstm":
            return EncDecOccLSTM
        
        elif model_class == "ed_lstm_exp":
            return EncDecOccLSTMExperimental

        else:
            raise ValueError("Model not supported.")
    
    ######## Initialization ########
    def initialize_all(self, train_dict:dict, val_dict:dict, test_dict:dict):
        
        train_set, val_set, test_set = self.initialize_dataset(
            train_dict, 
            val_dict, 
            test_dict
        )

        train_loader, val_loader, test_loader = self.initialize_dataloader(train_set, val_set, test_set)

        model = self.initialize_model()
        optimizer = self.initialize_optimizer(model)
        
        return train_loader, val_loader, test_loader, model, optimizer
    
    def initialize_model(self) -> nn.Module:
        
        occcount = "occcount" in self.hyperparameters["features"]
        self.update_hyperparameters(
            {"occcount": occcount}
            )
        
        model = self.model_class(self.hyperparameters, self.path_to_helpers)
        
        return model.to(self.device)    
    
    def initialize_optimizer(self, model:nn.Module) -> Optimizer:
        
        if self.optimizer_class == torch.optim.Adam:
            optimizer = self.optimizer_class(model.parameters(), lr=self.hyperparameters["lr"], 
                                             weight_decay=self.hyperparameters["weight_decay"])
            
        elif self.optimizer_class == torch.optim.SGD:
            optimizer = self.optimizer_class(model.parameters(), lr=self.hyperparameters["lr"], 
                                             momentum=self.hyperparameters["momentum"],
                                             weight_decay=self.hyperparameters["weight_decay"])
            
        else:
            raise ValueError("Optimizer not supported.")
        
        return optimizer
    
    def initialize_dataset(self, train_dict:dict, val_dict:dict, test_dict:dict):
        
        train_set = OccupancyDataset(train_dict, self.hyperparameters, self.path_to_helpers, validation=False)
        val_set = OccupancyDataset(val_dict, self.hyperparameters, self.path_to_helpers, validation=True)
        test_set = OccupancyDataset(test_dict, self.hyperparameters, self.path_to_helpers, validation=True)    
        
        _, X, y_features, y = train_set[0]

        self.update_hyperparameters({
            "x_size": int(X.shape[1]),
            "y_features_size": int(y_features.shape[1]), 
            "y_size": int(y.shape[1])}
        )
        
        return train_set, val_set, test_set

    def initialize_dataloader(self, train_set:Dataset, val_set:Dataset, test_set:Dataset):
        
        
        train_sampler = WeightedRandomSampler(train_set.sample_weights, len(train_set.samples), generator=self.torch_rng)       
        train_loader = DataLoader(train_set, batch_size=self.hyperparameters["batch_size"], 
                                  collate_fn=self.custom_collate, generator=self.torch_rng, sampler=train_sampler)
        
        val_loader = DataLoader(val_set, batch_size=self.hyperparameters["batch_size"], shuffle=False, 
                                collate_fn=self.custom_collate)
        
        
        test_loader = DataLoader(test_set, batch_size=self.hyperparameters["batch_size"], shuffle=False, 
                                 collate_fn=self.custom_collate)
        
        return train_loader, val_loader, test_loader
        
    ######## Training ########
    def train_one_epoch(self, dataloader:DataLoader, model:nn.Module, 
                        optimizer:Optimizer, val_dataloader:DataLoader=None):
            """
            trains a model for one epoch
            """
            self.stats_logger.reset_train_loss_buffer()
            
            
            
            #optimizer.zero_grad()
            #losses = []
            for info, X, y_features, y, room_id in dataloader:
                    
                optimizer.zero_grad()
                
                room_id = room_id.to(self.device)
                X = X.to(self.device)
                y_features = y_features.to(self.device)
                y = y.to(self.device).view(-1, model.output_size)

                model_output = model(X, y_features, room_id)
                #room_capa = torch.Tensor([x[-1] for x in info]).to(self.device)

                loss = self.criterion(model_output, y)
                #loss_i = self.criterion(model_output, y)
                
                #losses.append(loss_i)
                    
                #if len(losses) == 8:
                #    loss = torch.mean(torch.stack(losses))
                #    loss.backward()
                #    optimizer.step()
                #    losses = []
                #    optimizer.zero_grad()
                #    #self.writer.add_scalar("Loss/train", loss.cpu().detach().float(), self.n_updates)
                #    self.n_updates += 1
                #    self.stats_logger.append_train_loss(loss.cpu().detach().float())
                    
                loss.backward()
                optimizer.step()                   
                    
                self.n_updates += 1
                self.stats_logger.append_train_loss(loss.cpu().detach().float())
                
                if self.n_updates % 50 == 0:
                    self.summary_writer.add_scalar("Loss/train", np.mean(self.stats_logger.train_loss[self.n_updates-self.train_log_inteval:self.n_updates]), self.n_updates-self.train_log_inteval)
                
                
                # validate every 100 updates
                if ((self.n_updates % self.test_interval) == 0) and val_dataloader:
                    self.test_one_epoch(val_dataloader, model, log_info=True)
                    self.n_test += 1
                    self.summary_writer.add_scalar("Loss/val", np.mean(self.stats_logger.val_loss[self.n_test]), self.n_test*self.test_interval)
                    if np.mean(self.stats_logger.val_loss[self.n_test]) < self.best_loss:
                        self.best_loss = np.mean(self.stats_logger.val_loss[self.n_test])
                        model = model.to("cpu")
                        self.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            save_path=self.cp_path
                        )
                        model = model.to(self.device)
                
                if self.n_updates >= self.hyperparameters["max_n_updates"]:
                    break
    
    def train_n_updates(self, train_dataloader:DataLoader, val_dataloader:DataLoader, 
                        model:nn.Module, optimizer:Optimizer, log_predictions:bool):
        
        # Evaluate untrained model
        self.test_one_epoch(val_dataloader, model, log_info=True)
        self.summary_writer.add_scalar("Loss/val", np.mean(self.stats_logger.val_loss[0]), 0)

        # Calculate number of epochs from max_n_updates
        if self.hyperparameters["max_n_updates"] % len(train_dataloader) == 0:
            n_epochs = self.hyperparameters["max_n_updates"] // len(train_dataloader)
        else:
            n_epochs = self.hyperparameters["max_n_updates"] // len(train_dataloader) + 1

        n_epochs = int(n_epochs)
        for _ in tqdm.tqdm(range(n_epochs), leave=False):
                
            self.train_one_epoch(train_dataloader, model, optimizer, val_dataloader)
            
            if not log_predictions:
                self.stats_logger.free_memory()
            
                
            self.stats_logger.append_mean_train_loss()      
             
    ######## Validation ########
    def test_one_epoch(self, dataloader:DataLoader, model:nn.Module, log_info:bool):
        """
        validates a model for one epoch
        """

        model.eval()
        
        info_list = []
        val_loss = []
        predictions = []
        targets = []
        inputs = []
        
        with torch.no_grad():
            for info, X, y_features, y, room_id in dataloader:
                   
                room_id = room_id.to(self.device)
                X = X.to(self.device)
                y_features = y_features.to(self.device)
                y = y.to(self.device).view(-1, model.output_size)

                model_output = model(X, y_features, room_id)
                #room_capa = torch.Tensor([x[-1] for x in info]).to(self.device

                loss = self.criterion(model_output, y)

                val_loss.append(loss.cpu().detach())
                inputs.append((X.cpu().detach(), y_features.cpu().detach()))
                predictions.append(model_output.cpu().detach())
                targets.append(y.cpu().detach())
                info_list.append(info)
                
        model.train()
        if log_info:
            self.stats_logger.append_val_stats(val_loss, predictions, targets, inputs, info=info_list) 
        else:
            self.stats_logger.append_val_stats(val_loss, predictions, targets, inputs) 
        
    ##### Summary Writer ######
    def text_to_writer(self, hyperparameters:dict):
        self.summary_writer.add_text("Hyperparameters", json.dumps(hyperparameters, indent=4))

    def hyperparameters_to_writer(self, val_loss:float, train_loss:float):
        
        hyperparams_writer = self.hyperparameters.copy()
        hyperparams_writer["hidden_size"] = "_".join([str(x) for x in hyperparams_writer["hidden_size"]])
        hyperparams_writer["room_ids"] = "_".join([str(x) for x in hyperparams_writer["room_ids"]])
        hyperparams_writer["permissible_features"] = "_".join(hyperparams_writer["permissible_features"])
        
        self.summary_writer.add_hparams(
            hparam_dict=hyperparams_writer, 
            metric_dict={'hparam/val_loss_L1': float(val_loss),
                         'hparam/train_loss_L1': float(train_loss)},
            run_name=f"hparam",
            global_step = 0
        )
    #### Save and Load ####
    def save_checkpoint(self, model:nn.Module, optimizer:Optimizer, save_path:str=None):
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, "info.txt"), "w") as file:
            file.write(f"N Updates: {self.n_updates}\n")
            file.write(f"Best Loss: {self.best_loss}")
            
        #torch.save(train_loader.dataset, os.path.join(self.save_path, "train_dataset.pt"))
        #torch.save(val_loader.dataset, os.path.join(self.save_path, "val_dataset.pt"))
        #torch.save(test_loader.dataset, os.path.join(self.save_path, "test_dataset.pt"))
        torch.save(model.cpu().state_dict(), os.path.join(save_path, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        self.dict_to_json(save_path, self.hyperparameters, "hyperparameters")

    def save_hyperparameters(self, save_path:str):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.dict_to_json(save_path, self.hyperparameters, "hyperparameters")   
            
    def dict_to_json(self, path_to_dir, dictionary, file_name):
        with open(os.path.join(path_to_dir, f"{file_name}.json"), "w") as file:
            json.dump(dictionary, file, indent=4
                      )
        return None
    
    def load_checkpoint(self, checkpoint_path:str):
        
        # ignore warnings
        hyperparameters = json.load(open(os.path.join(checkpoint_path, "hyperparameters.json"), "r"))
        
        model = self.model_class(hyperparameters, self.path_to_helpers)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt"), weights_only=True))
        model = model.to(self.device)
        
        optimizer = self.optimizer_class(model.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), weights_only=True))
        
        #train_set = torch.load(os.path.join(checkpoint_path, "train_dataset.pt"))
        #val_set = torch.load(os.path.join(checkpoint_path, "val_dataset.pt"))
        #test_set = torch.load(os.path.join(checkpoint_path, "test_dataset.pt"))
        
        return model, optimizer,  hyperparameters

