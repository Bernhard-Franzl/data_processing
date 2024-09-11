
from _forecasting import OccupancyDataset
import torch
from _dfguru import DataFrameGuru as DFG

from testing import run_detailed_test, plot_predictions, prepare_model_and_data
from tqdm import tqdm
import os
import numpy as np
import time

def list_checkpoints(path_to_dir, run_id):
    
    path_to_run = os.path.join(path_to_dir, f"run_{run_id}")
    
    if os.path.exists(path_to_run):
        comb_ids = list(map(lambda x: int(x.split("_")[-1]), os.listdir(path_to_run)))
        run_comb_tuples = list(zip([run_id]*len(comb_ids), comb_ids))
        del comb_ids
        return run_comb_tuples
    
    else:
        raise ValueError(f"Checkpoints of run {run_id} do not exist")    
   
def run_n_tests(run_comb_tuples, cp_log_dir, mode, plot, data):
    
    dfg = DFG()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    list_combs = []
    dict_losses = {"MAE":[], "MSE":[], "RMSE":[], "R2":[]}

    bar_format = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    for n_run, n_comb in tqdm(run_comb_tuples, total=len(run_comb_tuples), bar_format=bar_format, leave=False):

        checkpoint_path = os.path.join(cp_log_dir, f"run_{n_run}", f"comb_{n_comb}")

        model, hyperparameters, data_dicts, room_ids = prepare_model_and_data(
            checkpoint_path=checkpoint_path, 
            dfg=dfg, 
            device=device)
        
        if data == "train":
            data_dict = data_dicts[0]
        elif data == "val":
            data_dict = data_dicts[1]
        elif data == "test":
            data_dict = data_dicts[2]
        else:
            raise ValueError("Data must be 'train', 'val' or 'test'")
        
        # load dataset
        dataset = OccupancyDataset(data_dict, hyperparameters, mode)

        # run detailed test
        losses, predictions = run_detailed_test(model, dataset, device)
        
        for key in dict_losses:
            dict_losses[key].append(losses[key])
        list_combs.append((n_run, n_comb))
        
        if plot:
            plot_predictions(dataset, predictions, room_ids, n_run, n_comb)
            
    return list_combs, dict_losses

def get_k_smallest_largest(k:int, losses:dict):
    
    smallest_k = torch.topk(torch.Tensor(losses), k, largest=False).indices
    largest_k = torch.topk(torch.Tensor(losses), k, largest=True).indices
    
    return smallest_k, largest_k

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
  
def erase_file(file_name):
    with open(file_name, "w") as file:
        file.write("")      
  
def write_new_line(file_name):
    with open(file_name, "a") as file:
        file.write("\n")

def evaluate_results(filename, list_combs, dict_losses):


    for key, value in dict_losses.items():
        
        mean_losses = np.array([torch.mean(torch.Tensor(x)) for x in value])
        # sort by mean loss, descending if R2 -> we sort best to worst
        if key == "R2":
            indices = np.argsort(mean_losses)[::-1]
        else:
            indices = np.argsort(mean_losses)
            
        write_loss_to_txt(filename, list_combs[indices], mean_losses[indices], key)

    write_new_line(filename)
    
    
cp_log_dir = "_forecasting/checkpoints"
mode = "dayahead"
filename = "results.txt"

##############################
## Run n tests
#for data in ["val", "train"]:
#    for run_id in [1, 2]:
    
#        tuples_run_comb = list_checkpoints(cp_log_dir, run_id)

#        list_combs, dict_losses = run_n_tests(
#            run_comb_tuples=tuples_run_comb,
#            cp_log_dir=cp_log_dir, 
#            mode=mode, 
#            plot=False, 
#            data=data)   
                
#        list_combs = np.array(list_combs)

#        write_header_to_txt(filename, run_id, data)

#        evaluate_results(filename, list_combs, dict_losses)


##############################
# Test chosen combinations
for data in ["val"]:
    
    list_combs, dict_losses = run_n_tests(
        run_comb_tuples=[(1,27),(1,6), (2,8), (2,9)],
        cp_log_dir=cp_log_dir, 
        mode=mode, 
        plot=True, 
        data=data
    )   
                

#mt.test_one_epoch(train_loader, model, log_info=True)
#mt.test_one_epoch(val_loader, model, log_info=True)

#losses = mt.stats_logger.val_loss
#predictions = mt.stats_logger.val_pred
#targets = mt.stats_logger.val_target
#inputs = mt.stats_logger.val_input
#infos = mt.stats_logger.val_info


#import numpy as np
#import pandas as pd

#for i, type in enumerate(["train", "val"]):

#    pred_i = torch.cat(predictions[i], dim=0)
#    target_i = torch.cat(targets[i], dim=0)
    
#    y_times = []
#    room_ids = []   
#    for x in infos[i]:
#        #room_id, x_time, y_time, features, room_capa = x
#        for room_id, x_time, y_time, _, room_capa in x:
#            y_times.append(y_time.values[0])
#            room_ids.append(room_id)
    
    
#    data = pd.DataFrame(columns=["time", "pred", "target", "room_id"],
#                        data={"time":y_times, "pred":pred_i.squeeze(), "target":target_i.squeeze(), "room_id":room_ids})

#    plot_data = data[data["room_id"]==0].sort_values(by="time")

#    fig = go.Figure()
    
#    fig.add_trace(
#        go.Scatter(
#            x=plot_data["time"],
#            y=plot_data["pred"],
#            mode="lines+markers",
#            name="Prediction"
#        )
#    )
    
#    fig.add_trace(
#        go.Scatter(
#            x=plot_data["time"],
#            y=plot_data["target"],
#            mode="lines+markers",
#            name="Target"
#        )
#    )
    
#    fig.show()

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
            




