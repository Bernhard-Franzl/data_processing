import numpy as np
from _forecasting import list_checkpoints, run_n_tests, write_header_to_txt, evaluate_results, evaluate_results_lecture

  
cp_log_dir = "_forecasting/checkpoints/presence"
#mode = "normal"
#filename = f"results_{mode}.txt"

##############################
# Run n tests

#for data in ["train", "val"]:
#    for mode in ["dayahead", "unlimited"]:
#        filename = f"results_{mode}_presence.txt"
#        for run_id in [0]:
            
#            tuples_run_comb = list_checkpoints(cp_log_dir, run_id)

#            list_combs, dict_losses, list_hyperparameters = run_n_tests(
#                run_comb_tuples=tuples_run_comb,
#                cp_log_dir=cp_log_dir, 
#                mode=mode, 
#                plot=False, 
#                data=data)   
                    
#            list_combs = np.array(list_combs)

#            write_header_to_txt(filename, run_id, data)

#            evaluate_results(filename, list_combs, dict_losses, list_hyperparameters, top_k_params=5)


##############################
# Test chosen combinations
#import torch
#for mode in ["dayahead", "unlimited"]:
#    filename = f"results_{mode}.txt"
#    for data in ["train", "val", "test"]:
        
#        list_combs, dict_losses, list_hyperparameters = run_n_tests(
#            run_comb_tuples=[(0,4)],
#            cp_log_dir=cp_log_dir, 
#            mode=mode, 
#            plot=True,  
#            data=data
#        )   
        
#        # claculate mean losses
#        print(f"---------- {data} --------------")
#        for  i in range(len(list_combs)):
#            print(f"Combination: {list_combs[i]}")
#            for key in dict_losses:
#                mean_loss = torch.mean(torch.Tensor(dict_losses[key][i]))
#                print(f"Mean {key} loss: {mean_loss}")
#            print(f"------------------------")

###############################  Lecture Dataset ########################################

#cp_log_dir = "_forecasting/checkpoints/lecture"
#for data in ["train"]:
#    for mode in ["onedateahead"]:
#        filename = f"results_lecture_{mode}.txt"
#        for run_id in [0]:
            
#            tuples_run_comb = list_checkpoints(cp_log_dir, run_id)

#            list_combs, dict_losses, list_hyperparameters = run_n_tests(
#                run_comb_tuples=tuples_run_comb,
#                cp_log_dir=cp_log_dir, 
#                mode=mode, 
#                plot=False, 
#                data=data)   
                    
#            list_combs = np.array(list_combs)

#            write_header_to_txt(filename, run_id, data)
#            evaluate_results_lecture(filename, list_combs, dict_losses, list_hyperparameters, top_k_params=5)




# Test chosen combinations
import torch
cp_log_dir = "_forecasting/checkpoints/lecture"
for mode in ["onedateahead"]:
    filename = f"results_lecture_{mode}.txt"
    for data in ["train", "val"]:
        
        list_combs, dict_losses, list_hyperparameters = run_n_tests(
            run_comb_tuples=[(4,0)],
            cp_log_dir=cp_log_dir, 
            mode=mode, 
            plot=True,  
            data=data
        )   
        
        # claculate mean losses
        print(f"---------- {data} --------------")
        for  i in range(len(list_combs)):
            print(f"Combination: {list_combs[i]}")
            for key in dict_losses:
                if key == "R2":
                    continue
                mean_loss = torch.mean(torch.Tensor(dict_losses[key][i]))
                print(f"Mean {key} loss: {mean_loss}")
            print(f"------------------------")




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
            




