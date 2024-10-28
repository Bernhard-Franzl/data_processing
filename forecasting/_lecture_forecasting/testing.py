import torch
import torchmetrics
from torch.optim import Adam
from torch.utils.data import DataLoader

import os
import json
import time

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np
from tqdm import tqdm

from _lecture_forecasting.model import SimpleLectureDenseNet, SimpleLectureLSTM
from _lecture_forecasting.data import  LectureDataset
from _lecture_forecasting.data import load_data

from torch.nn.utils.rnn import pad_sequence

from _dfguru import DataFrameGuru as DFG

class TestWriter():
    
    def __init__(self, filename, erase_file):
        
        self.set_filename(filename)
        if erase_file:
            self.erase_file()
    
    ########## Basic functions ##########
    def set_filename(self, filename):
        self.filename = filename
        
    def erase_file(self ):
        with open(self.filename, "w") as file:
            file.write("") 

    def write_new_line(self):
        with open(self.filename, "a") as file:
            file.write("\n")
            
    def write_text_to_file(self, text):
        with open(self.filename, "a") as file:
            file.write(text)
            
    ############## Advanced functions ##############
    def write_header(self, run_id):
        
        header_txt = f"################# Run: {run_id} #################\n"
        header_txt += f"Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}\n"
        self.write_text_to_file(header_txt)
        
    def write_loss(self, loss_f, data, combinations, losses, losses_std, baseline_losses, bs_losses_std):            
        
        loss_txt = f"Loss function: {loss_f} | Data: {data}\n"
        loss_txt += f"Combinations: {combinations.tolist()}\n"
        loss_txt += f"Losses: {losses.tolist()}\n"
        loss_txt += f"Baseline losses: {baseline_losses.tolist()}\n"
        loss_txt += f"Losses std: {losses_std.tolist()}\n"
        loss_txt += f"Baseline losses std: {bs_losses_std.tolist()}\n"
        
        self.write_text_to_file(loss_txt)
        
    def write_hyperparameters(self, hyperparameters):
            
        listy = [(key, hyperparameters[key])  for  key in sorted(hyperparameters.keys())]
        hyperparameters_txt = f"Hyperparameters: {listy}\n"
        self.write_text_to_file(hyperparameters_txt)


class LectureTestSuite():
    
    def __init__(self, cp_log_dir, path_to_data, path_to_helpers, path_to_results, erase_results_file):
        
        self.cp_log_dir = cp_log_dir
        self.dfg = DFG()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #self.loss_types = ["MAE", "MSE", "RMSE"]
        self.loss_types = ["MAE"]
        self.loss_functions = self.get_loss_functions()
        
        self.bar_format = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        
        self.path_to_data = path_to_data
        self.path_to_helpers = path_to_helpers
        
        self.hyperparameters = None
        
        self.writer = TestWriter(path_to_results, erase_results_file)
    
    def get_loss_functions(self):
        
        loss_functions = dict()
        if "MAE" in self.loss_types:
            loss_functions["MAE"] = torch.nn.L1Loss(reduction="none")
        if "MSE" in self.loss_types:
            loss_functions["MSE"] = torch.nn.MSELoss(reduction="none")
        if "RMSE" in self.loss_types:
            loss_functions["RMSE"] = lambda x, y: torch.sqrt(torch.nn.MSELoss(reduction="none")(x, y))
            
        return loss_functions
    
    
    ########## Get list of checkpoints ##########    
    def list_checkpoints(self, run_id):
    
        path_to_run = os.path.join(self.cp_log_dir, f"run_{run_id}")
        
        if os.path.exists(path_to_run):
            comb_ids = list(map(lambda x: int(x.split("_")[-1]), os.listdir(path_to_run)))
            run_comb_tuples = list(zip([run_id]*len(comb_ids), comb_ids))
            del comb_ids
            return run_comb_tuples
        
        else:
            raise ValueError(f"Checkpoints of run {run_id} do not exist")  
      
    ########## Load checkpoints ##########
    def handle_model_class(self, model_name:str):

        if model_name == "simple_lecture_lstm":
            return SimpleLectureLSTM
        
        elif model_name == "simple_lecture_densenet":
            return SimpleLectureDenseNet
        
        else:
            raise ValueError(f"Model {model_name} not recognized")
        
    def load_checkpoint(self, checkpoint_path:str, load_optimizer:bool, path_to_helpers:str):
        
        self.hyperparameters = json.load(open(os.path.join(checkpoint_path, "hyperparameters.json"), "r"))
        
        # ignore warnings    
        model_class = self.handle_model_class(self.hyperparameters["model_class"])
        model = model_class(self.hyperparameters, path_to_helpers)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt"), weights_only=True))
        
        if load_optimizer:
            optimizer = Adam(model.parameters(), lr=self.hyperparameters["lr"], weight_decay=self.hyperparameters["weight_decay"])
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), weights_only=True))
            return model, optimizer

        return model
    
    def prepare_data(self, split_by:str, dataset_mode:str, path_to_helpers:str):
        
        if dataset_mode in ["time_sequential", "time_onedateahead"]:
        
            train_df, val_df, test_df = load_data(
                self.path_to_data, 
                dfguru=self.dfg,
                split_by=split_by)
            

            trainset = LectureDataset(
                lecture_df=train_df, 
                hyperparameters=self.hyperparameters, 
                dataset_mode=dataset_mode, 
                path_to_helpers=path_to_helpers,
                validation=False
            )
            valset = LectureDataset(
                lecture_df=val_df, 
                hyperparameters=self.hyperparameters, 
                dataset_mode=dataset_mode, 
                path_to_helpers=path_to_helpers,
                validation=True
            )
            testset = LectureDataset(
                lecture_df=test_df, 
                hyperparameters=self.hyperparameters, 
                dataset_mode=dataset_mode, 
                path_to_helpers=path_to_helpers,
                validation=True
            )
            
            return trainset, valset, testset
                
        else:
            raise ValueError(f"Mode {dataset_mode} not recognized")  
    
    def sequential_collate(self, x):
        
        info = [x_i[0] for x_i in x]
        X = pad_sequence([x_i[1] for x_i in x], batch_first=True, padding_value=self.hyperparameters["padding_value"], padding_side="left")
        #X = torch.stack([x_i[1] for x_i in x])
        y_features = torch.stack([x_i[2] for x_i in x])
        y = torch.stack([x_i[3] for x_i in x])
        immutable_features = torch.stack([x_i[0][6] for x_i in x])

        return info, X, y_features, y, immutable_features
    
    def dateahead_collate(self, x):
        
        info = [x_i[0] for x_i in x]
        X = torch.stack([x_i[1] for x_i in x])
        y_features = torch.stack([x_i[2] for x_i in x])
        y = torch.stack([x_i[3] for x_i in x])
        immutable_features = torch.stack([x_i[0][6] for x_i in x])
                          
        return info, X, y_features, y, immutable_features
    
    def prepare_data_loader(self, dataset, dataset_mode):
        
        if dataset_mode == "time_onedateahead":
            collate_f = self.dateahead_collate
        elif dataset_mode == "time_sequential":
            collate_f = self.sequential_collate
        else:
            raise ValueError("Dataset mode not supported.")
        
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                  collate_fn=collate_f)
        return data_loader
    
    
    def calculate_losses(self, losses, pred, target):

        for key, loss_f in self.loss_functions.items():
            
            loss = loss_f(pred, target)
            
            if loss.shape[0] == 1:
                losses[key].append(loss.squeeze())
                
            else:
                losses[key].extend(loss_f(pred, target).squeeze())
        
    def test_naive_baseline(self, dataset, losses):
        
        # simply predicts the last observed value
        
        predictions = []
        infos = []
        inputs = []
        targets = []
        target_features = []
        
        #for info, X, y_features, y in tqdm(dataset, total=len(dataset), bar_format=self.bar_format, leave=False):
        for info, X, y_features, y in dataset:
            
            if len(X.shape) == 1:
                preds = X[:1]
                
                if preds < 0:
                    preds = torch.Tensor([0])
                

            elif len(X.shape) == 2:
                occrates = X[:, :1]
                occrates_pos = occrates[occrates >= 0]
                
                preds = occrates_pos[-1:]
                y = y.squeeze(-1)
                if preds.shape[0] == 0:
                    preds = torch.Tensor([0])
                
            else:
                raise ValueError("X has unknown shape")
            
            if any(preds < 0):
                print(preds)
                raise ValueError("Negative value in prediction")
            
            
            if preds.shape != y.shape:
                print(X, preds, y)
                raise ValueError("Prediction and target shape do not match")
            
        
            self.calculate_losses(losses, preds, y)
            
            predictions.append(preds)
            infos.append(info)
            inputs.append(X)
            targets.append(y)
            target_features.append(y_features)

        return losses, predictions, infos, targets, inputs, target_features
 
    def test_avg_baseline(self, dataset, losses):
        
        # simply predicts the last observed value
        
        predictions = []
        infos = []
        inputs = []
        targets = []
        target_features = []
        
        #for info, X, y_features, y in tqdm(dataset, total=len(dataset), bar_format=self.bar_format, leave=False):
        for info, X, y_features, y in dataset: 
            
            if len(X.shape)== 1:
                preds = X[:1]
                raise ValueError("X has strange shape")
                
            elif len(X.shape) == 2:
                occrates = X[:, :1]
                occrates_pos = occrates[occrates >= 0]
            
                if occrates_pos.shape[0] == 0:
                    occrates_pos = torch.Tensor([0])
                
                preds = torch.mean(occrates_pos, dim=0, keepdim=True)

            else:
                raise ValueError("X has unknown shape")
            
            if any(preds < 0):
                print(preds)
                raise ValueError("Negative value in prediction")
            
            y = y.squeeze(-1)
            
            if preds.shape != y.shape:
                print(X, preds, y)
                raise ValueError("Prediction and target shape do not match")
            
        
            self.calculate_losses(losses, preds, y)
            
            predictions.append(preds)
            infos.append(info)
            inputs.append(X)
            targets.append(y)
            target_features.append(y_features)

        return losses, predictions, infos, targets, inputs, target_features
       
    def test_model(self, model, dataloader, losses):
        
        model.eval()
        model = model.to(self.device)

        with torch.no_grad():
            
            predictions = []
            infos = []
            inputs = []
            targets = []
            target_features = []

            #for info, X, y_features, y, immutable_features  in tqdm(dataloader, total=len(dataloader), bar_format=self.bar_format, leave=False):
            for info, X, y_features, y, immutable_features in dataloader:
                
                immutable_features = immutable_features.to(self.device)
                X = X.to(self.device)
                y_features = y_features.to(self.device)
                y = y.to("cpu").view(-1, model.output_size)
            
                model_output = model(X, y_features, immutable_features).cpu()
                
                self.calculate_losses(losses, model_output, y)

                predictions.extend(model_output)
                infos.extend(info)
                inputs.extend(X.cpu())
                targets.extend(y)
                target_features.extend(y_features.cpu())
        
        return losses, predictions, infos, targets, inputs, target_features
    
    def denormalize_predictions(self, list_of_predictions, targets, infos, dataset, convert_to_counts):

        if convert_to_counts:
            
            min_registered = dataset.min_registered
            max_registered = dataset.max_registered
            
            lecture_df = dataset.lec_df
            registered = []
            occcount_ori = []
            info_save = []
            
            for info in infos:
                mask = (lecture_df["coursenumber"] == info[0]) & (lecture_df["starttime"] == info[2]) & (lecture_df["roomid"] == info[5])
                

                registered.append(dataset.lec_df[mask]["registered"].values[0])
                info_save.append((info[0], info[2], info[5]))
                #occrate_ori.append(dataset.lec_df[mask]["occrate_ori"].values[0])
                #occrate.append(dataset.lec_df[mask]["occrate"].values[0])
                occcount_ori.append(dataset.lec_df[mask]["occcount"].values[0])

            registered_denorm = (np.array(registered) * (max_registered - min_registered)) + min_registered

        
        min_occrate = dataset.min_occrate
        max_occrate = dataset.max_occrate
        
        list_denorm = []
        for predictions in list_of_predictions:
            pred_denorm = (np.array(predictions).squeeze() * (max_occrate - min_occrate)) + min_occrate
            
            if convert_to_counts:
                pred_denorm = pred_denorm * registered_denorm
                
            list_denorm.append(pred_denorm.squeeze())
        

        target_denorm = (np.array(targets).squeeze() * (max_occrate - min_occrate)) + min_occrate
        if convert_to_counts:
            target_denorm = target_denorm * registered_denorm
        
        return tuple(list_denorm), target_denorm
                
                
    def evaluate_combinations(self, comb_tuples, split_by, plot_results):
        
        convert_to_counts = False
        
        list_combinations = []
        list_hyperparameters = []
        list_loss_dicts = []
        list_baseline_loss_dicts = []
        list_avg_baseline_loss_dicts = []
        list_dataset_masks = []
        
        for n_run, n_comb in tqdm(comb_tuples, total=len(comb_tuples), bar_format=self.bar_format, leave=False):

            checkpoint_path = os.path.join(self.cp_log_dir, f"run_{n_run}", f"comb_{n_comb}")
            pth = os.path.join(self.path_to_helpers, f"helpers_lecture_{split_by}.json")
            
            model = self.load_checkpoint(checkpoint_path=checkpoint_path, load_optimizer=False, path_to_helpers=pth)
            
            list_hyperparameters.append(self.hyperparameters)
            list_combinations.append((n_run, n_comb))
            
            if "dataset_mode" not in self.hyperparameters:
                dataset_mode = "time_sequential"
            else:
                dataset_mode = self.hyperparameters["dataset_mode"]
                
            trainset, valset, testset = self.prepare_data(
                split_by=split_by,
                dataset_mode=dataset_mode,
                path_to_helpers=pth
            )
            
            ###################################################
                
            dict_losses= {key:[] for key in self.loss_types}
            dict_baseline_losses = {key:[] for key in self.loss_types}
            dict_avg_losses = {key:[] for key in self.loss_types}
            
            
            # for plotting
            list_predictions = []
            list_infos = []
            list_targets = []
            list_baseline_predictions = []
                    
            for dataset in [trainset, valset, testset]:

        
                dataloader = self.prepare_data_loader(
                    dataset=dataset,
                    dataset_mode=dataset_mode,
                )

                dict_baseline_losses, bl_preds, bl_infos, bl_targets, bl_inputs, bl_target_features = self.test_naive_baseline(dataset, dict_baseline_losses)

                #dict_avg_losses, avg_preds, avg_infos, avg_targets, avg_inputs, avg_target_features = self.test_avg_baseline(dataset, dict_avg_losses)
                
                dict_losses, predictions, infos, targets, inputs, target_features = self.test_model(model, dataloader, dict_losses)

                # denormalize the prediction
                
                (predictions, bl_preds), targets = self.denormalize_predictions(
                    [predictions, bl_preds], 
                    targets, 
                    infos, 
                    dataset, 
                    True)

                
                list_predictions.extend(predictions)
                list_infos.extend(infos)
                list_targets.extend(targets)
                list_baseline_predictions.extend(bl_preds)

                
            list_baseline_loss_dicts.append(dict_baseline_losses)
            list_avg_baseline_loss_dicts.append(dict_avg_losses)
            
            list_loss_dicts.append(dict_losses)
            
            dataset_mask = np.array([0]*len(trainset) + [1]*len(valset) + [2]*len(testset)) 
            
            if plot_results:
                
                import matplotlib.pyplot as plt
                #info = (lecture_id, X_df["starttime"], y_df["starttime"], self.exogenous_features, X_df["roomid"], y_df["roomid"], lecture_immutable_features)
                
                pred_array = np.array(list_predictions).squeeze()
                pred_baseline = np.array(list_baseline_predictions).squeeze()
                target_array = np.array(list_targets).squeeze()
                
                mae = torch.nn.L1Loss(reduction="mean")
                mse = torch.nn.MSELoss(reduction="mean")
                
                val_mask = (dataset_mask == 1)
                print(f"N_RUN: {n_run} | N_COMB: {n_comb}")
                print(f"MAE: {mae(torch.Tensor(pred_array[val_mask]), torch.Tensor(target_array[val_mask]))}")
                print(f"MSE: {mse(torch.Tensor(pred_array[val_mask]), torch.Tensor(target_array[val_mask]))}")
                
                
                y_start_time = np.array([x[2] for x in list_infos]).squeeze()
                y_room_id = np.array([x[5] for x in list_infos]).squeeze()
                
                
                room_id_list = [0,1]    

                fig = make_subplots(rows=2, cols=1,
                                    vertical_spacing = 0.2,
                                    subplot_titles=[f"Room {room_id}" for room_id in room_id_list],)
                
                #fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                
                #room 0
                for room_id in [0, 1]:

                    # plot validation set
                    mask = (y_room_id == room_id) & (dataset_mask == 1)
                    

                    y_start_time_room = y_start_time[mask]
                    pred_array_room = pred_array[mask]
                    target_array_room = target_array[mask]
                    pred_baseline_room = pred_baseline[mask]
                    
                    sort_indices = np.argsort(y_start_time_room)
                    y_start_time_room = y_start_time_room[sort_indices]
                    
                    pred_array_room = pred_array_room[sort_indices]      
                    target_array_room = target_array_room[sort_indices] 
                    pred_baseline_room = pred_baseline_room[sort_indices]  
                            
                    if room_id == 0:
                        legend_name = "legend"
                    else:
                        legend_name = f"legend{room_id}"
                        
                    fig.add_trace(
                        go.Bar(
                            x=y_start_time_room,
                            y=pred_baseline_room,
                            name=f"Baseline Model",
                            legend=f"legend{room_id+1}"
                        ),
                        row=room_id+1, col=1
                    )
                    fig.add_trace(
                        go.Bar(
                            x=y_start_time_room,
                            y=target_array_room,
                            name=f"Targets",
                            legend=f"legend{room_id+1}"
                        ),
                        row=room_id+1, col=1
                    )
                    fig.add_trace(
                        go.Bar(
                            x=y_start_time_room,
                            y=pred_array_room,
                            name=f"Predictions Model",
                            legend=f"legend{room_id+1}"
                        ),
                        row=room_id+1, col=1
                    )
                
                # add title
                fig.update_layout(
                    title=dict(
                        text=f"Run {n_run} - Combination {n_comb}", 
                        font=dict(size=30), 
                        y = 0.95,    
                        x = 0.5,
                        xanchor = 'center',
                        yanchor = 'top'
                    )
                )
                
                fig.update_layout(
                    margin=dict( l = 100, r = 200, b = 100, t = 150),
                    height=1000,
                    legend1=dict(
                        title="Room 0",
                        yanchor="auto",
                        y=1,
                        xanchor="right",
                        x=1.1
                    ),
                    legend2=dict(
                        title="Room 1",
                        yanchor="auto",
                        y=0.35,
                        xanchor="right",
                        x=1.1
                    )
                )
                fig.show()
            
            list_dataset_masks.append(dataset_mask)
            
        return list_combinations, list_hyperparameters, list_loss_dicts, list_baseline_loss_dicts, list_dataset_masks

    def analyse_results(self, combinations, hyperparameters, loss_dicts, baseline_loss_dicts, dataset_masks):
        
        for loss_f in self.loss_types:
            
            mean_train_loss = []
            mean_val_loss = []
            
            mean_bs_train_loss = []
            mean_bs_val_loss = []
            
            std_train_loss = []
            std_val_loss = []
            std_bs_train_loss = []
            std_bs_val_loss = []
        
            for i, (n_run, n_comb) in enumerate(combinations):

                loss_dict_i = loss_dicts[i]
                baseline_loss_dicts_i = baseline_loss_dicts[i]
                dataset_mask_i = dataset_masks[i]
                
                losses_i = np.array(loss_dict_i[loss_f])
                baseline_losses_i = np.array(baseline_loss_dicts_i[loss_f])
                
                train_losses_i = losses_i[dataset_mask_i == 0]
                val_losses_i = losses_i[dataset_mask_i == 1]
                
                train_bs_losses_i = baseline_losses_i[dataset_mask_i == 0]
                val_bs_losses_i = baseline_losses_i[dataset_mask_i == 1]
                
                mean_train_loss.append(np.mean(train_losses_i))
                mean_val_loss.append(np.mean(val_losses_i))
                mean_bs_train_loss.append(np.mean(train_bs_losses_i))
                mean_bs_val_loss.append(np.mean(val_bs_losses_i))
                
                std_train_loss.append(np.std(train_losses_i))
                std_val_loss.append(np.std(val_losses_i))
                std_bs_train_loss.append(np.std(train_bs_losses_i))
                std_bs_val_loss.append(np.std(val_bs_losses_i))
            
            
            # train losses
            indices = np.argsort(mean_train_loss)
            self.writer.write_loss(
                loss_f=loss_f,
                data="train",
                combinations=np.array(combinations)[indices],
                losses=np.array(mean_train_loss)[indices], losses_std=np.array(std_train_loss)[indices],
                baseline_losses=np.array(mean_bs_train_loss)[indices], bs_losses_std=np.array(std_bs_train_loss)[indices]
            )
            
            # val losses
            indices = np.argsort(mean_val_loss)
            self.writer.write_loss(
                loss_f=loss_f,
                data="val",
                combinations=np.array(combinations)[indices],
                losses=np.array(mean_val_loss)[indices], losses_std=np.array(std_val_loss)[indices],
                baseline_losses=np.array(mean_bs_val_loss)[indices], bs_losses_std=np.array(std_bs_val_loss)[indices]
            )
                
            self.writer.write_new_line()
    
    
    
    
    
