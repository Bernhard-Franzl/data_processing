import torch
import torchmetrics
from torch.optim import Adam
from torch.utils.data import DataLoader

import os
import json
import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import numpy as np
from tqdm import tqdm
import pandas as pd


from _occupancy_forecasting.model import SimpleOccDenseNet, SimpleOccLSTM, EncDecOccLSTM #,SimpleOccGRU, MassConservingOccLSTM
from _occupancy_forecasting.data import OccupancyDataset
from _occupancy_forecasting.data import load_data

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

class OccupancyTestSuite():
    
    def __init__(self, cp_log_dir, path_to_data, path_to_helpers, path_to_results, erase_results_file):
        
        self.cp_log_dir = cp_log_dir
        self.dfg = DFG()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.loss_types = ["MAE", "R2"]
        self.loss_functions = self.get_loss_functions()
        
        self.bar_format = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        
        self.path_to_data = path_to_data
        self.path_to_helpers = path_to_helpers
        
        self.hyperparameters = None
        
        self.writer = TestWriter(path_to_results, erase_results_file)
    
    def get_loss_functions(self):
        
        loss_functions = dict()
        if "MAE" in self.loss_types:
            loss_functions["MAE"] = torch.nn.L1Loss(reduction="mean")
        if "MSE" in self.loss_types:
            loss_functions["MSE"] = torch.nn.MSELoss(reduction="mean")
        if "RMSE" in self.loss_types:
            loss_functions["RMSE"] = lambda x, y: torch.sqrt(torch.nn.MSELoss(reduction="mean")(x, y))
        if "R2" in self.loss_types:
            loss_functions["R2"] = torchmetrics.R2Score()
              
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

        
        if model_name == "simple_densenet":
            return SimpleOccDenseNet
        
        elif model_name == "simple_lstm":
            return SimpleOccLSTM
        
        elif model_name == "ed_lstm":
            return EncDecOccLSTM
        
        else:
            raise ValueError(f"Model {model_name} not recognized")
        
    def load_checkpoint(self, checkpoint_path:str, load_optimizer:bool):

        # ignore warnings    
        model_class = self.handle_model_class(self.hyperparameters["model_class"])
        model = model_class(self.hyperparameters, self.path_to_helpers)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt"), weights_only=True))
        
        if load_optimizer:
            optimizer = Adam(model.parameters(), lr=self.hyperparameters["lr"], weight_decay=self.hyperparameters["weight_decay"])
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), weights_only=True))
            return model, optimizer

        return model
    
    def prepare_data(self, dataset_mode):
        
        if dataset_mode in ["normal", "dayahead", "unlimited"]:
                
            train_dict, val_dict, test_dict = load_data(
                path_to_data_dir=self.path_to_data, 
                frequency=self.hyperparameters["frequency"], 
                split_by=self.hyperparameters["split_by"],
                dfguru=self.dfg
            )
                    
            
            train_set = OccupancyDataset(train_dict, self.hyperparameters, self.path_to_helpers, validation=True)
            val_set = OccupancyDataset(val_dict, self.hyperparameters, self.path_to_helpers, validation=True)
            test_set = OccupancyDataset(test_dict, self.hyperparameters, self.path_to_helpers, validation=True)    

            return train_set, val_set, test_set
                
        else:
            raise ValueError(f"Mode {dataset_mode} not recognized")  
    
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
    
    def prepare_data_loader(self, dataset):
   
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,  
                                  collate_fn=self.custom_collate, drop_last=True)
        
        return dataloader
    
    
    def calculate_losses(self, losses, pred, target):

        for key, loss_f in self.loss_functions.items():
            
            if key == "R2":
                
                pred_s = pred.squeeze()
                if len(pred_s) == 1:
                    raise
                    losses[key].append(None)
                else:
                    loss = loss_f(pred_s, target.squeeze())
                    losses[key].append(loss)
                    
            else:
                loss = loss_f(pred, target)
                losses[key].append(loss)
        
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
            for info, X, y_features, y, additional_info in dataloader:
                
                additional_info = additional_info.to(self.device)
                X = X.to(self.device)
                y_features = y_features.to(self.device)
                y = y.to("cpu")#.view(-1, model.output_size)
     
                model_output = model.forecast_iter(X, y_features, y.shape[1], additional_info).cpu()

                if len(model_output) == 0:
                    continue
                
                y_adjusted = y[:, :model_output.shape[1]].squeeze(-1)
                y_features[:, :model_output.shape[1]]
                
                if len(info) != 1:
                    raise ValueError("Info has wrong shape")
                
                info = [(info[0][0], info[0][1], info[0][2][:model_output.shape[1]], info[0][3], info[0][4], info[0][5])]
                
                  
                self.calculate_losses(losses, model_output, y_adjusted)
                
                predictions.extend(model_output)
                infos.extend(info)
                inputs.extend(X.cpu())
                targets.extend(y_adjusted)
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
                         
    def evaluate_combinations(self, comb_tuples, plot_results, print_results, dataset_mode):
        
        convert_to_counts = False
        
        list_combinations = []
        list_hyperparameters = []
        list_loss_dicts = []
        list_baseline_loss_dicts = []
        list_avg_baseline_loss_dicts = []
        list_dataset_masks = []
        
        for n_run, n_comb in tqdm(comb_tuples, total=len(comb_tuples), bar_format=self.bar_format, leave=False):
            
            checkpoint_path = os.path.join(self.cp_log_dir, f"run_{n_run}", f"comb_{n_comb}")
            
            
            self.hyperparameters = json.load(open(os.path.join(checkpoint_path, "hyperparameters.json"), "r"))
        
            #self.pth = os.path.join(self.path_to_helpers, f"helpers_occpred.json")
            
            model = self.load_checkpoint(checkpoint_path=checkpoint_path, load_optimizer=False)
            
            list_hyperparameters.append(self.hyperparameters)
            list_combinations.append((n_run, n_comb))


            # handle dataset mode
            if dataset_mode == None:
                dataset_mode = self.hyperparameters["dataset_mode"]
            else:
                dataset_mode = dataset_mode
                self.hyperparameters["dataset_mode"] = dataset_mode
            
            trainset, valset, testset = self.prepare_data(
                dataset_mode=dataset_mode
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
                    dataset=dataset
                )

                #dict_baseline_losses, bl_preds, bl_infos, bl_targets, bl_inputs, bl_target_features = self.test_naive_baseline(dataset, dict_baseline_losses)
                #dict_avg_losses, avg_preds, avg_infos, avg_targets, avg_inputs, avg_target_features = self.test_avg_baseline(dataset, dict_avg_losses)
                
                dict_losses, predictions, infos, targets, inputs, target_features = self.test_model(model, dataloader, dict_losses)
                
                # denormalize the prediction
                #(predictions, bl_preds), targets = self.denormalize_predictions(
                #    [predictions, bl_preds], 
                #    targets, 
                #    infos, 
                #    dataset, 
                #    True)

                
                list_predictions.extend(predictions)
                list_infos.extend(infos)
                list_targets.extend(targets)
                #list_baseline_predictions.extend(bl_preds)

                
            list_baseline_loss_dicts.append(dict_baseline_losses)
            list_avg_baseline_loss_dicts.append(dict_avg_losses)
            
            list_loss_dicts.append(dict_losses)
            
            dataset_mask = np.array([0]*len(trainset) + [1]*len(valset) + [2]*len(testset)) 
            list_dataset_masks.append(dataset_mask)
            
            
            if print_results:
                
                mae_losses = np.array(dict_losses["MAE"])
                r2_losses = np.array(dict_losses["R2"])
                
                val_mask = (dataset_mask == 1)
                train_mask = (dataset_mask == 0)
                test_mask = (dataset_mask == 2)
                
                print(f"N_RUN: {n_run} | N_COMB: {n_comb}")
                
                # without baseline
                print(f"Train: MAE: {np.round(np.mean(mae_losses[train_mask]) ,4)} | R2: {np.round(np.mean(r2_losses[train_mask]) ,4)}")
                print(f"Val: MAE: {np.round(np.mean(mae_losses[val_mask]) ,4)} | R2: {np.round(np.mean(r2_losses[val_mask]) ,4)}")
                print(f"Test: MAE: {np.round(np.mean(mae_losses[test_mask]) ,4)} | R2: {np.round(np.mean(r2_losses[test_mask]) ,4)}")

                
                # with baseline
                #val_preds = pred_array[val_mask]
                #val_targets = target_array[val_mask]
                #val_bl_preds = pred_baseline[val_mask]
                #print(f"MAE Val: {np.round(mae(torch.Tensor(val_preds), torch.Tensor(val_targets)).numpy(), 4)}| BL: {np.round(mae(torch.Tensor(val_bl_preds), torch.Tensor(val_targets)).numpy(), 4)}")
                
                #train_preds = pred_array[train_mask]
                #train_targets = target_array[train_mask]
                #train_bl_preds = pred_baseline[train_mask]
                #print(f"MAE Train: {np.round(mae(torch.Tensor(train_preds), torch.Tensor(train_targets)).numpy(), 4)}| BL: {np.round(mae(torch.Tensor(train_bl_preds), torch.Tensor(train_targets)).numpy(), 4)}")
                
                #test_preds = pred_array[test_mask]
                #test_targets = target_array[test_mask]
                #test_bl_preds = pred_baseline[test_mask]
                #print(f"MAE Test: {np.round(mae(torch.Tensor(test_preds), torch.Tensor(test_targets)).numpy(), 4)}| BL: {np.round(mae(torch.Tensor(test_bl_preds), torch.Tensor(test_targets)).numpy(), 4)}")
            
                
            if plot_results:
                
                #info = (lecture_id, X_df["starttime"], y_df["starttime"], self.exogenous_features, X_df["roomid"], y_df["roomid"], lecture_immutable_features)
                #pred_array = np.array(list_predictions).squeeze()
                #pred_baseline = np.array(list_baseline_predictions).squeeze()
                #target_array = np.array(list_targets).squeeze()

                if dataset_mode=="dayahead":
                    
                    room_id_list = [0,1]
                    dataset_id_list = [0, 1, 2]
                    dataset_string = ["Train", "Val", "Test"]
                    subplot_titles = [f"Room {room_id} - Data {data_id}" for room_id in room_id_list for data_id in dataset_id_list]
                    fig = make_subplots(rows=2, cols=3,
                                        column_widths=[0.5, 0.25, 0.25],
                                        vertical_spacing = 0.2,
                                        subplot_titles=subplot_titles)
                    
                    y_room_id = np.array([x[0] for x in list_infos])
                    
                    color_sequence = px.colors.qualitative.Plotly

                    
                    for room_id in room_id_list:
                        for dataset_id in dataset_id_list:
                            
                            mask = (dataset_mask == dataset_id) & (y_room_id == room_id)
                            
                            masked_list_pred = [list_predictions[i] for i in range(len(list_predictions)) if mask[i]]
                            masked_list_target = [list_targets[i] for i in range(len(list_targets)) if mask[i]]
                            masked_list_y_time = [list_infos[i][2] for i in range(len(list_infos)) if mask[i]]
                            
                            pred_array = np.concatenate(masked_list_pred)
                            time_array = np.concatenate(masked_list_y_time)
                            target_array = np.concatenate(masked_list_target)

                            dataset_array = np.full(len(pred_array), dataset_id)
                            

                            df_plot = pd.DataFrame(
                                {"Time": time_array,
                                "Pred": pred_array,
                                "Target": target_array,
                                "Color": dataset_array}
                            )
                            
                            name = f"Target-{dataset_string[dataset_id]}-{room_id}"
                            fig.add_trace(
                                go.Scatter(
                                    x=df_plot["Time"],
                                    y=df_plot["Target"],
                                    mode="markers+lines",
                                    name=name,
                                    marker=dict(color=color_sequence[dataset_id]),
                                    line=dict(color=color_sequence[dataset_id]),
                                ),
                                row=room_id+1, col=dataset_id+1
                            )
                            
                            name = f"Prediction-{dataset_string[dataset_id]}-{room_id}"
                            fig.add_trace(
                                go.Scatter(
                                    x=df_plot["Time"],
                                    y=df_plot["Pred"],
                                    mode="markers+lines",
                                    name=name,
                                    marker=dict(color=color_sequence[dataset_id+1]),
                                    line=dict(color=color_sequence[dataset_id+1]),
                                ),
                                row=room_id+1, col=dataset_id+1
                            )
                          
                    # general layout  
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
                    )
                    fig.show()

                    
                else:
                                        
                    val_mask = (dataset_mask == 1)
                    train_mask = (dataset_mask == 0)
                    test_mask = (dataset_mask == 2)
                    
                    y_start_time = np.array([x[2] for x in list_infos])
                    y_room_id = np.array([x[0] for x in list_infos]) 
                    
                    room_id_list = [0,1]    
                    
                    fig = make_subplots(rows=2, cols=1,
                                        vertical_spacing = 0.2,
                                        subplot_titles=[f"Room {room_id}" for room_id in room_id_list],)
                    
                    #fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                        
                    #room 0
                    for room_id in room_id_list:
                        
                                        
                        #for data_id in [0, 1, 2]:

                        # plot validation set
                        mask = y_room_id == room_id #& (dataset_mask == data_id)
                        
                    
                        y_start_time_room = y_start_time[mask]
                        

                        # sanity check
                        
                        if not (len(y_start_time_room) == mask.sum()):
                            raise ValueError("Not all timestamps are unique")
                        raise
                        pred_array_room = pred_array[mask]
                        target_array_room = target_array[mask]
                        
                        raise
                        #pred_baseline_room = pred_baseline[mask]
                        dataset_array_room = dataset_mask[mask]
                        

                        sort_indices = np.argsort(y_start_time_room[:, 0])
                        
                        y_start_time_room = y_start_time_room[sort_indices]
                        
                        pred_array_room = pred_array_room[sort_indices]      
                        target_array_room = target_array_room[sort_indices] 
                        #pred_baseline_room = pred_baseline_room[sort_indices]  
                        dataset_array_room = dataset_array_room[sort_indices]
                        
                        if dataset_mode == "normal":
                            pred_array_room = pred_array_room[:, 0]
                            target_array_room = target_array_room[:, 0]
                            y_start_time_room = y_start_time_room[:, 0]
                            #pred_baseline_room = pred_baseline_room[:, 0]
                            

                        df_room = pd.DataFrame(
                            {"Time": y_start_time_room,
                            "Pred": pred_array_room,
                            "Target": target_array_room,
                            "Color_Target": dataset_array_room,
                            "Index": np.arange(len(y_start_time_room))}
                        )
                        #type_mapping = {0: "Train", 1: "Val", 2: "Test"}

                        #df_room["Color_Target"] = df_room["Color_Target"].map(type_mapping)
                        #df_room["Color_Pred"] = df_room["Color_Target"] + "_Pred"
                        
                        df_room["Color_Pred"] = df_room["Color_Target"] + 0.5
                        
                        fig.add_trace(
                            px.scatter(
                                data_frame=df_room,
                                x="Index",
                                y="Target",
                                color="Color_Target",
                                render_mode="webgl",
                            )["data"][0],
                            row=room_id+1, col=1
                        )
                        fig.add_trace(
                            px.scatter(
                                data_frame=df_room,
                                x="Index",
                                y="Pred",
                                color="Color_Pred",
                                render_mode="webgl",
                            )["data"][0],
                            row=room_id+1, col=1
                        )
                        
    
                    #fig.update_traces(width=0.5)
                        
                    #    fig.add_trace(
                    #        go.Bar(
                    #            x=y_start_time_room,
                    #            y=pred_baseline_room,
                    #            name=f"Baseline Model",
                    #            legend=f"legend{room_id+1}"
                    #        ),
                    #        row=room_id+1, col=1
                    #    )
                    #    fig.add_trace(
                    #        go.Bar(
                    #            x=y_start_time_room,
                    #            y=target_array_room,
                    #            name=f"Targets",
                    #            legend=f"legend{room_id+1}"
                    #        ),
                    #        row=room_id+1, col=1
                    #    )
                    #    fig.add_trace(
                    #    go.Bar(
                    #        x=y_start_time_room,
                    #        y=pred_array_room,
                    #        name=f"Predictions Model",
                    #        legend=f"legend{room_id+1}"
                    #    ),
                    #    row=room_id+1, col=1
                    #)
                
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
                
        return list_combinations, list_hyperparameters, list_loss_dicts, list_baseline_loss_dicts, list_dataset_masks


    def write_data_loss(self, data_string, loss_f, combinations, hyperparameters, mean_loss, std_loss, mean_bl_loss, std_bl_loss, skip_baseline):
        
        if loss_f == "R2":
            indices = np.argsort(mean_loss)[::-1]
        else:
            indices = np.argsort(mean_loss)
            
            
        top_k_params=5
        
        list_hyperparameters_k = [hyperparameters[i] for i in indices[:top_k_params]]
        all_keys = all_keys = set().union(*list_hyperparameters_k)

        param_results = {}
        for key in all_keys:
            vc = np.unique([params_dict[key] for params_dict in list_hyperparameters_k], return_counts=True, axis=0)
            # make vc readable
            vc = list(zip(vc[0], vc[1]))
            param_results[key] = vc
          
        if skip_baseline:
            bs_loss = np.empty(0)
            bs_loss_std = np.empty(0)
        else:
            bs_loss = np.array(mean_bl_loss)[indices]
            bs_loss_std = np.array(std_bl_loss)[indices]
            
        self.writer.write_loss(
            loss_f=loss_f,
            data=data_string,
            combinations=np.array(combinations)[indices],
            losses=np.array(mean_loss)[indices], losses_std=np.array(std_loss)[indices],
            baseline_losses=bs_loss, bs_losses_std=bs_loss_std
        )
        self.writer.write_hyperparameters(param_results)
        self.writer.write_new_line()
               
    def analyse_results(self, combinations, hyperparameters, loss_dicts, 
                            baseline_loss_dicts, dataset_masks):
        
        for loss_f in self.loss_types:
            
            mean_train_loss = []
            mean_val_loss = []
            mean_test_loss = []
            
            mean_bs_train_loss = []
            mean_bs_val_loss = []
            mean_bs_test_loss = []
            
            std_train_loss = []
            std_val_loss = []
            std_test_loss = []
            std_bs_train_loss = []
            std_bs_val_loss = []
            std_bs_test_loss = []
            
            skip_baseline = False
            for i, (n_run, n_comb) in enumerate(combinations):

                loss_dict_i = loss_dicts[i]
                dataset_mask_i = dataset_masks[i]
                
                losses_i = np.array(loss_dict_i[loss_f])
                
                train_losses_i = losses_i[dataset_mask_i == 0]
                val_losses_i = losses_i[dataset_mask_i == 1]
                test_losses_i = losses_i[dataset_mask_i == 2]


                mean_train_loss.append(np.mean(train_losses_i))
                mean_val_loss.append(np.mean(val_losses_i))
                mean_test_loss.append(np.mean(test_losses_i))
                
                std_train_loss.append(np.std(train_losses_i))
                std_val_loss.append(np.std(val_losses_i))
                std_test_loss.append(np.std(test_losses_i))
                        
                # baseline stuff        
                baseline_loss_dicts_i = baseline_loss_dicts[i]
                baseline_losses_i = np.array(baseline_loss_dicts_i[loss_f])
        
                if len(baseline_losses_i) == 0:
                    skip_baseline = True
                    continue
                    
                else:
                    train_bs_losses_i = baseline_losses_i[dataset_mask_i == 0]
                    val_bs_losses_i = baseline_losses_i[dataset_mask_i == 1]
                                
                    mean_bs_train_loss.append(np.mean(train_bs_losses_i))
                    mean_bs_val_loss.append(np.mean(val_bs_losses_i))
                    std_bs_train_loss.append(np.std(train_bs_losses_i))
                    std_bs_val_loss.append(np.std(val_bs_losses_i))
                    raise

            
            # write train losses
            self.write_data_loss(
                data_string="train",
                loss_f=loss_f,
                combinations=combinations,
                hyperparameters=hyperparameters,
                mean_loss=mean_train_loss,
                std_loss=std_train_loss,
                mean_bl_loss=mean_bs_train_loss,
                std_bl_loss=std_bs_train_loss,
                skip_baseline=skip_baseline
            )
            
            # write val losses
            self.write_data_loss(
                data_string="val",
                loss_f=loss_f,
                combinations=combinations,
                hyperparameters=hyperparameters,
                mean_loss=mean_val_loss,
                std_loss=std_val_loss,
                mean_bl_loss=mean_bs_val_loss,
                std_bl_loss=std_bs_val_loss,
                skip_baseline=skip_baseline
            )
            
            # write val losses
            self.write_data_loss(
                data_string="test",
                loss_f=loss_f,
                combinations=combinations,
                hyperparameters=hyperparameters,
                mean_loss=mean_test_loss,
                std_loss=std_test_loss,
                mean_bl_loss=mean_bs_test_loss,
                std_bl_loss=std_bs_test_loss,
                skip_baseline=skip_baseline
            )
                
        self.writer.write_new_line()
        self.writer.write_new_line()
    
    
    
    
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
        
        elif model_name == "ed_lstm":
            return EncDecOccLSTM
        
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
     
    if mode in ["normal", "dayahead", "unlimited"]:
        
        train_dict, val_dict, test_dict = load_data(
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
        
    elif mode in ["time_sequential", "time_onedateahead"]:
        
        #train_df, val_df, test_df = load_data_lecture("data", dfguru=dfg)
        
        #if data == "train":
        #    data_dict = train_df
        #    validation = False
        #elif data == "val":
        #    data_dict = val_df
        #    validation = True
        #elif data == "test":
        #    data_dict = test_df
        #    validation = True
        #else:
        #    raise ValueError(f"Data {data} not recognized")
        
        
        #dataset = LectureDataset(data_dict, hyperparameters, mode, validation)
        #room_ids = None
        pass
        #y_samples = []
        #for i in range(len(dataset)):
        #    y_samples.append(dataset[i][3])
            
    else:
        raise ValueError(f"Mode {mode} not recognized")
    
    
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
        if not(info[5][0] == None):
            X_course = info[5][0].to(torch.int32)
            y_course = info[5][1].to(torch.int32)
            room_id = torch.cat([X_course, y_course]).to(device)
            
        else:
            room_id = None
            
        y_features = y_features.to(device)
        
        with torch.no_grad():
            
            preds = model.forecast_iter(X, y_features, len(y), room_id)
                    
            if len(preds) == 0:
                continue
            
            preds = preds.to("cpu")
            y_adjusted = y[:len(preds)]
            y_features = y_features[:len(preds)]
            
            
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

def run_detailed_test_forward(model, dataset:OccupancyDataset, device):
    
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
        
        X = X.to(device)[None, :]
        if dataset.dataset_mode == "time_sequential":
            room_id = info[6].to(device)[None, :]
        y_features = y_features.to(device)[None, :]
        
        with torch.no_grad():
            
            preds = model(X, y_features, room_id)
            
            #if len(preds) == 0:
            #    continue
            
            preds = preds.to("cpu")
            #y_adjusted = y[:len(preds)]
            
            if model.discretization:
                preds = torch.argmax(preds, dim=-1).to(dtype=torch.float32)
                y = torch.argmax(y, dim=-1).to(dtype=torch.float32)
            
            info = (info[0], info[1], info[2], info[3], info[4], info[5], info[6])
                
            losses["MAE"].append(mae_f(preds, y))
            losses["MSE"].append(mse_f(preds, y))
            losses["RMSE"].append(torch.sqrt(mse_f(preds, y)))
            if len(preds) == 1:
                losses["R2"].append(None)
            else:
                losses["R2"].append(r2_f(preds, y))
            
        predictions.append(preds)
        infos.append(info)
        inputs.append(X)
        targets.append(y)
        target_features.append(y_features)
    
    return losses, predictions, infos, targets, inputs, target_features

def run_naive_baseline_lecture(model, dataset:OccupancyDataset, device):
    # simply predicts the last observed value
    
    
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
        
        #print("output_size:", model.output_size)
        
        if len(X.shape)== 1:
            preds = X[:model.output_size]
        elif len(X.shape) == 2:
            preds = X[-1:, :model.output_size]
        else:
            raise ValueError("X has unknown shape")

        if model.discretization:
            preds = torch.argmax(torch.nn.functional.softmax(preds), dim=-1).to(dtype=torch.float32)
            y = torch.argmax(y, dim=-1).to(dtype=torch.float32)
        
        losses["MAE"].append(mae_f(preds, y))
        losses["MSE"].append(mse_f(preds, y))
        losses["RMSE"].append(torch.sqrt(mse_f(preds, y)))
            
        predictions.append(preds)
        infos.append(info)
        inputs.append(X)
        targets.append(y)
        target_features.append(y_features)

    return losses, predictions, infos, targets, inputs, target_features
    
def run_n_tests(run_comb_tuples, cp_log_dir, mode, plot, data, naive_baseline):
    
    dfg = DFG()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    list_combs = []
    dict_losses = {"MAE":[], "MSE":[], "RMSE":[], "R2":[]}
    baseline_losses = {"MAE":[], "MSE":[], "RMSE":[], "R2":[]}
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

        naive_preds = None
        # run detailed test
        if mode in ["normal", "dayahead", "unlimited"]:
            losses, predictions, infos, targets, inputs, target_features = run_detailed_test(model, dataset, device)
        else:
            if naive_baseline:
                naive_losses, naive_preds, _, _, _, _ = run_naive_baseline_lecture(model, dataset, device)
                
                for key in baseline_losses:
                    baseline_losses[key].append(naive_losses[key])
                    
            losses, predictions, infos, targets, inputs, target_features = run_detailed_test_forward(model, dataset, device)
        
        for key in dict_losses:
            dict_losses[key].append(losses[key])
        
        list_combs.append((n_run, n_comb))

        losses_mae = np.array(losses["MAE"])
        argsort_losses = np.argsort(losses_mae)[::-1]
        greater_0 = argsort_losses[losses_mae[argsort_losses] > 0]
        #print(argsort_losses)
        
        if plot:
            if mode in ["normal", "dayahead", "unlimited"]:
                plot_predictions(infos, predictions, targets, room_ids, n_run, n_comb, target_features)
            else:
                plot_predictions_lecture(infos, predictions, targets, room_ids, n_run, n_comb, naive_preds)
            
    if naive_baseline:
        return list_combs, dict_losses, list_hyperparameters, baseline_losses
    else:
        return list_combs, dict_losses, list_hyperparameters

def plot_predictions(infos:list, predictions:list, targets:list, room_ids:list, n_run:int, n_comb:int, target_features:list, naive_predictions=None):
    
    dict_y_times = dict([(room_id,[]) for room_id in room_ids])    
    dict_preds = dict([(room_id,[]) for room_id in room_ids])
    dict_targets = dict([(room_id,[]) for room_id in room_ids])
    dict_target_features = dict([(room_id,[]) for room_id in room_ids])
    
    
    for i, pred in enumerate(predictions):

            room_id = infos[i][0]
            y_time = infos[i][2]
            
            pred = pred.numpy()
            
            dict_y_times[room_id].append(y_time.values)

            dict_target_features[room_id].append(target_features[i].cpu().numpy())

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
        
        target_feature_names = infos[0][3]
        target_features_room = dict_target_features[room_id]
        x=np.concatenate(dict_y_times[room_id])
        y=np.concatenate(target_features_room) 
        
        for i in range(len(target_feature_names)):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y[:, i],
                    mode="lines+markers",
                    name=f"{target_feature_names[i]} Room {room_id}"
                )
            )
        
        
        fig.update_layout(
            title=f"Run {n_run} - Combination {n_comb} - Room {room_id}",
            xaxis_title="Time",
            yaxis_title="Occupancy"
        )
        
        fig.show()
        
def plot_predictions_lecture(infos:list, predictions:list, targets:list, room_ids:list, n_run:int, n_comb:int, naive_predictions:list=None):
    
    #dict_y_times = dict([(room_id,[]) for room_id in room_ids])    
    #dict_preds = dict([(room_id,[]) for room_id in room_ids])
    #dict_targets = dict([(room_id,[]) for room_id in room_ids])
    
    
    predictions = [x.squeeze().numpy() for x in predictions]
    targets = [x.squeeze().numpy() for x in targets]
    x_axis = np.arange(len(predictions))
        
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=predictions,
            mode="lines+markers",
            name=f"Predictions"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=targets,
            mode="lines+markers",
            name=f"Targets"
        )
    )
    if naive_predictions:
        naive_predictions = [x.squeeze().numpy() for x in naive_predictions]
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=naive_predictions,
                mode="lines+markers",
                name=f"Naive Baseline"
            )
        )
    
    fig.update_layout(
        title=f"Run {n_run} - Combination {n_comb}",
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
    
def write_loss_to_txt(file_name, combinations, losses, baseline_losses, loss_f):
    
    with open(file_name, "a") as file:
        file.write(f"Loss function: {loss_f}\n")
        file.write(f"Combinations: {combinations.tolist()}\n")
        file.write(f"Losses: {losses.tolist()}\n")
        file.write(f"Baseline losses: {baseline_losses.tolist()}\n")
  
def write_loss_to_txt_without_baseline(file_name, combinations, losses, loss_f):
        
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
def evaluate_results(filename, list_combs, dict_losses, list_hyperparameters, baseline_losses, top_k_params):
    
    for key, value in dict_losses.items():
        
        mean_losses = np.array([torch.mean(torch.Tensor(x)) for x in value])
        if not (baseline_losses == None):
            mean_baseline_losses = np.array([torch.mean(torch.Tensor(x)) for x in baseline_losses[key]])
        
        # sort by mean loss, descending if R2 -> we sort best to worst
        if key == "R2":
            indices = np.argsort(mean_losses)[::-1]
        else:
            indices = np.argsort(mean_losses)
            

        if (baseline_losses == None):
            write_loss_to_txt_without_baseline(filename, list_combs[indices], mean_losses[indices], key)
        else:
            write_loss_to_txt(filename, list_combs[indices], mean_losses[indices], mean_baseline_losses[indices], key)

        
        
        list_hyperparameters_k = [list_hyperparameters[i] for i in indices[:top_k_params]]
        all_keys = all_keys = set().union(*list_hyperparameters_k)
        param_results = {}
        for key in all_keys:
            vc = np.unique([str(params_dict[key]) for params_dict in list_hyperparameters_k], return_counts=True, axis=0)
            # make vc readable
            vc = list(zip(vc[0], vc[1]))
            param_results[key] = vc
          
        write_hyperparameters_to_txt(filename, param_results)  
        
    write_new_line(filename)
   
def evaluate_results_lecture(filename, list_combs, dict_losses, list_hyperparameters, baseline_losses, top_k_params):

    for key, value in dict_losses.items():
        
    
        # sort by mean loss, descending if R2 -> we sort best to worst
        if key == "R2":
            continue
            mean_losses = np.array([torch.mean(torch.Tensor(x)) for x in value])
            indices = np.argsort(mean_losses)[::-1]
        else:
            mean_losses = np.array([torch.mean(torch.Tensor(x)) for x in value])
            mean_baseline_losses = np.array([torch.mean(torch.Tensor(x)) for x in baseline_losses[key]])
            indices = np.argsort(mean_losses)

            
        write_loss_to_txt(filename, list_combs[indices], mean_losses[indices], mean_baseline_losses[indices], key)
        
        #list_hyperparameters_k = [list_hyperparameters[i] for i in indices[:top_k_params]]
        #all_keys = all_keys = set().union(*list_hyperparameters_k)
        #param_results = {}
        #for key in all_keys:
        #    vc = np.unique([params_dict[key] for params_dict in list_hyperparameters_k], return_counts=True, axis=0)
        #    # make vc readable
        #    vc = list(zip(vc[0], vc[1]))
        #    param_results[key] = vc
          
        #write_hyperparameters_to_txt(filename, param_results)  
        
    write_new_line(filename)
    
def get_k_smallest_largest(k:int, losses:dict):
    
    smallest_k = torch.topk(torch.Tensor(losses), k, largest=False).indices
    largest_k = torch.topk(torch.Tensor(losses), k, largest=True).indices
    
    return smallest_k, largest_k