from preprocessing import Preprocessor, SignalPreprocessor
from utils import ParameterSearch, Evaluator
import numpy as np
import json
import os
import pandas as pd


#########  Constants #########
room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}
data_path = "/home/franzl/data_06_06/archive"

#TODO:
# incorporate paramter search into the Evaluator class
# implement a method that can be used to evaluate the results of the parameter search inside Evaluator class

# can we somehow implement a rule based approach when to take which prediction mode

# implement something like 75% percentile in calcl partiicpants
# before large test run check if results can be read in again!!



# test single parameter set
path_to_json = "signal_processing/parameters.json"
#params = json.load(open(path_to_json, "r"))
#print(params)

#preprocessor = SignalPreprocessor(data_path, room_to_id, door_to_id)
#cleaned_data, raw_data = preprocessor.apply_preprocessing(params)

#se_list, ae_list, ctd_list, se_min_list, ae_min_list = Evaluator("SignalAnalyzer", 
#                                       "data/zählung.csv").evaluate_signal_analyzer(data=cleaned_data,
#                                                                                    raw_data=raw_data, 
#                                                                                    params=params,
#                                                                                    details=True)
                                       
#for i, ae_min_entry in enumerate(ae_min_list):
#    mode, value, control_in, control_pred, participants, df = ae_min_entry
#    diff = abs(np.diff(participants))
#    describe = df["people_inside"].describe()
#    std = describe["std"]
#    diff = abs(np.diff(participants))
#    diff_ratio = diff/max(participants)
#    if diff_ratio >= 0.3:
#        desired_mode = "max"
#        #print(mode, value, std, diff, "Control:",control_in, " Preds:", control_pred, describe["50%"], round(describe["mean"],2), describe["75%"], " ##### ",participants, describe["std"])
    
#    elif diff_ratio <= 0.1:
#        desired_mode = "mean"
#        #print(i, mode, value, std, diff, diff_ratio, "Control:",control_in, " Preds:", describe["50%"], describe["75%"], " ###### ", round(describe["mean"],2), control_pred,  " ##### ",participants, describe["std"])

#    else:
#        print(i, mode, value, std, diff, diff_ratio, "Control:",control_in, " Preds:", describe["50%"], describe["75%"], " ###### ", round(describe["mean"],2), control_pred,  " ##### ",participants, describe["std"])

# Results: 
# Samples index 9 is not so good -> we miss some signals, but algrotithm is not so bad, its just the data
# Sample index 13 is no so good -> but its due to not ideal choice of the sample time point -> some students left earlier! CTD is low!


def write_results_to_txt(file_name, comb_number, params, se_list, ae_list, ctd_list):
    with open(file_name, "a") as file:
        file.write(f"######## Combination: {comb_number} ########\n")
        file.write(f"Parameters: {params}\n")
        file.write(f"MSE: {np.round(np.mean(se_list), 4)} MedianSE: {np.median(se_list)}\n")
        file.write(f"MAE: {np.round(np.mean(ae_list), 4)} MedianAE: {np.median(ae_list)}\n")
        file.write(f"MCTD: {np.round(np.mean(ctd_list), 4)} MedianCTD: {np.median(ctd_list)}\n")
        file.write("\n")
        file.write("\n")
        
    return None

def write_results_to_json(file_name, params, se_list, ae_list, ctd_list):
    
    with open(f"results/{file_name}.json", "w") as file:
        json.dump({"parameters":params, 
                   "SE":se_list, 
                   "AE":ae_list, 
                   "CTD":ctd_list}, file)
    
        
    return None


####### Parameter Search ########
path_to_json = "signal_processing/parameters.json"
comb_iterator = ParameterSearch(path_to_json=path_to_json).combinations_iterator(tqdm_bar=True)

preprocessor = SignalPreprocessor(data_path, room_to_id, door_to_id)

for i, params in enumerate(comb_iterator):
    
    if i == 0:
        answer = input("Are you sure you want to start? Have you checked file names?")
        if answer == "y":
            pass
        else:
            raise 
    
    cleaned_data, raw_data = preprocessor.apply_preprocessing(params)

    se_list, ae_list, ctd_list = Evaluator("SignalAnalyzer", "data/zählung.csv").evaluate_signal_analyzer(data=cleaned_data,
                                                                                                          raw_data=raw_data, 
                                                                                                          params=params)

    file_name = "results_time-window_finish.txt"
    write_results_to_txt(file_name, i, params, se_list, ae_list, ctd_list)
    file_name = f"comb_time-window_{i}"
    write_results_to_json(file_name, params, se_list, ae_list, ctd_list)
    

# Analyze the results
#parent_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = parent_dir.split("/signal_processing")[0]
#path_to_results = "results_2_10_6_1900"

#directories = [x for x in list(os.walk(parent_dir))[0][1] if "results" in x]

#se_list = []
#ae_list = []
#ctd_list = []
#parameters_list = []
#for directory in directories:
#    files = list(os.walk(os.path.join(parent_dir, directory)))[0][2]
#    for i, file in enumerate(files):
#        with open(f"{directory}/{file}", "r") as file:
#            results = json.load(file)
            
#            parameters_list.append(results["parameters"])
#            se_list.append(results["SE"])
#            ae_list.append(results["AE"])
#            ctd_list.append(results["CTD"])


#dataframe = pd.DataFrame({"parameters":parameters_list, "se":se_list, "ae":ae_list, "ctd":ctd_list})
#dataframe["mse"] = dataframe["se"].apply(lambda x: np.mean(x))
#dataframe["mae"] = dataframe["ae"].apply(lambda x: np.mean(x))
#dataframe["mctd"] = dataframe["ctd"].apply(lambda x: np.mean(x))


#sort_by_list= [["mae", "mse", "mctd"], ["mse", "mae", "mctd"], ["mctd", "mae", "mse"]]


#dict_list = []
#for  sort_by in sort_by_list:
#    dataframe_sorted = dataframe.sort_values(by=sort_by)

#    parameter_series_list = []
#    for i,row in iter(dataframe_sorted[:50].iterrows()):
#        parameter_series = pd.json_normalize(row["parameters"], sep="-")
#        parameter_series_list.append(parameter_series)
#        #print(f"######## Combination: {i} ########")
#        #print(row["mse"], row["mae"], row["mctd"])
        
#    parameters_df = pd.concat(parameter_series_list, axis=0)
#    #unique_values = [list(parameters_df[col].unique()) for col in parameters_df.columns]

#    #dict_uniqe_params = dict(list(zip(parameters_df.columns, unique_values)))
#    #print(dict_uniqe_params)
    
    
#    results_dict = dict()
#    for key in parameters_df.keys():
#        vcs = parameters_df[key].value_counts()
#        results_dict[key] = list(vcs.items())
#        file_name = "-".join(sort_by)
#        with open(f"results_{file_name}.json", "w") as file:
#            json.dump(results_dict, file, indent=4)


########  Data Preprocessing #########       


# cleaned_data = Preprocessor(data_path, room_to_id, door_to_id).apply_preprocessing(preprocessing_params)
# cleaned_data.to_csv("data/cleaned_data.csv")

# #cleaned_data = pd.read_csv("data/cleaned_data.csv", header=0, index_col=0)
# #cleaned_data["time"] = pd.to_datetime(cleaned_data["time"])


# #########  Data Analysis #########
# # evaluate methods of the SignalAnalyzer class
# # manual control data

# manual_control = pd.read_csv("data/zählung.csv")
# manual_control["room_id"] = manual_control["room"].map(room_to_id)
# mse = 0
# ae = 0
# for index, row in manual_control.iterrows():
#     room_id = row["room_id"]

#     start_time_int = row["start_time"]
#     end_time_int = row["end_time"]
    
#     time_string = row["time"].split(":")
    
#     date = row["date"].split(".")
#     day = int(date[0])
#     month = int(date[1])
    
#     control_people_in = row["people_in"]
    
#     control_time = dt(2024, month, day, int(time_string[0]), int(time_string[1]), 0)
#     start_time = dt(2024, month, day, start_time_int//100, start_time_int%100, 0)
#     end_time = dt(2024, month, day, end_time_int//100, end_time_int%100, 0)
    
#     first = bool(row["first"])
#     last = bool(row["last"])
    
#     analyzer = SignalAnalyzer()
#     data_analysis = analyzer.filter_by_room(cleaned_data, room_id)
    
#     delta = timedelta(minutes=30)
#     save_df = analyzer.filter_by_time(data_analysis, start_time-delta, end_time+delta)
    
    
#     # m is an extremely important parameter -> the one that is used to calculate the extrema
#     df_list, participants, extrema, df_list_plotting, control = analyzer.calc_participants(data_analysis, 
#                                             start_time=start_time,
#                                             end_time=end_time,
#                                             first=first,
#                                             last=last,
#                                             control=True,
#                                             mode="median")
    
    
#     df_plotting = analyzer.merge_participant_dfs(df_list_plotting)
    
#     horizontal_lines = [(participants[0], "black", " before"),
#                         (participants[1], "gold", " after")]

#     vertical_lines = [(control_time, "green", "start"),]
    
#     title = f"Control:{control_people_in}, Time:{control_time}"    
#     analyzer.plot_participants_algo(file_name = f"{control_time}_{room_id}.png",
#                              participants=df_plotting,
#                              df_list = None,
#                              control = None,
#                              extrema = None,
#                              horizontal_lines=[],
#                              vertical_lines=vertical_lines,
#                              title=title)

#     control_row = df_plotting[df_plotting["time"] == control_time]
    
#     #prediction = control_row["people_inside"].values[0]
#     #Mode: Mean 
#     #MSE:  88.55172413793103
#     #AE:  4.0
#     #Mode: Median
#     #MSE: 88.55172413793103
#     #AE: 4.0
    
#     #prediction = int(np.mean(participants))
#     # Mode: Mean
#     #MSE:  32.10344827586207
#     #AE:  3.68965517241379
#     #Mode: Median
#     #MSE: 31.689655172413794
#     #AE:  3.6206896551724137
    
#     prediction = int(np.max(participants))
#     # Mode: Mean
#     #MSE:  32.10344827586207
#     #AE:  3.68965517241379
#     #Mode: Median
#     #MSE:  11.96551724137931
#     #AE:  2.793103448275862
    
#     #prediction = int(np.min(participants))
#     #Mode: Mean, filter
#     #MSE:  100.89655172413794
#     #AE:  5.0344827586206895
#     #Mode: Median
#     #MSE:  105.10344827586206
#     #AE:  5.24137931034482
    
#     # try first of participants
#     #prediction = participants[0]
#     # try second of participants
#     #prediction = participants[1]
    
#     mse_term = (control_people_in - prediction)**2
#     mse += mse_term
#     ae_term = abs(control_people_in - prediction)
#     ae += ae_term
    
    
#     if mse_term > 10:
#         print("##################")
#         print("Time: ", control_time)
#         print("Room: ", room_id)
#         print("Participants: ", participants)
#         print("Control: ", control_people_in)
#         print("Prediction: ", prediction)
#         print("MSE: ", mse_term)
#         print("AE: ", ae_term)
#         print("##################")
#         print()

# print("MSE: ", mse/len(manual_control))
# print("AE: ", ae/len(manual_control))