from _preprocessing import SignalPreprocessor
from _evaluating import ParameterSearch, Evaluator, write_results_to_json, write_results_to_txt
from _dfguru import DataFrameGuru as DFG

import os
import json
import pandas as pd
import numpy as np

##TODO:
## Try different frequency values -> 0.5 minutes, 1 minute, 5 minutes etc. -> only relevant for calc participants
## incorporate the read results methods into the Evaluator class
## in evaluator add option to save results to json
## in evaluator add option to use stored preprocessed data

########  Constants #########
room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}
data_path = "/home/berni/Dropbox/data_29_06_merged/archive"
dfg = DFG()

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

###### Parameter Search ########

#path_to_json = "_preprocessing/parameters/parameters_discard_or_not.json"

#comb_iterator = ParameterSearch(path_to_json=path_to_json).combinations_iterator(tqdm_bar=True)

#preprocessor = SignalPreprocessor(data_path, room_to_id, door_to_id)

#for i, params in enumerate(comb_iterator):
    
#    if i == 0:
#        answer = input("Are you sure you want to start? Have you checked file names?")
#        if answer == "y":
#            pass
#        else:
#            raise 
        
#    #if (not params["filtering_params"]["handle_5"]) and (not params["filtering_params"]["handle_6"]):
#    #    continue
    
#    #if params["filtering_params"]["handle_5"] and params["filtering_params"]["handle_6"]:
#    #    continue
    
#    #if params["filtering_params"]["handle_5"] and (not params["filtering_params"]["handle_6"]):
#    #    continue

#    cleaned_data, raw_data = preprocessor.apply_preprocessing(params)
    
#    corrected_data = preprocessor.correct_25_04_HS19(cleaned_data, "/home/franzl/github_repos/data_processing/data/logs_25_04_HS19.txt")
    
#    ctd_list = Evaluator("PLCount", "/home/franzl/github_repos/data_processing/data/control_data/zählung.csv").evaluate_pl_count(data=corrected_data,
#                                                                                            dfguru=dfg,
#                                                                                            raw_data=raw_data, 
#                                                                                            params=params, 
#                                                                                            print_details=False)
#    file_name = "_preprocessing/results/results_time-window_discard.txt"
#    write_results_to_txt(file_name, i, params, ctd_list)
#    file_name = f"_preprocessing/results/discard/comb_time-window_{i}.json"
#    write_results_to_json(file_name, params, ctd_list)


############## Analyze Results ##############

###### test single parameter set ########


path_to_json = "_preprocessing/parameters/parameters_best.json"
params = json.load(open(path_to_json, "r"))

preprocessor = SignalPreprocessor(data_path, room_to_id, door_to_id)

cleaned_data, raw_data = preprocessor.apply_preprocessing(params)

corrected_data = preprocessor.correct_25_04_HS19(cleaned_data, "/home/berni/github_repos/data_processing/data/logs_25_04_HS19.txt")

ctd_list, control_details, pl_count_list, raw_data_list = Evaluator("PLCount", "/home/berni/github_repos/data_processing/data/control_data/zählung.csv").evaluate_pl_count(data=corrected_data,
                                                                                        dfguru=dfg,
                                                                                        raw_data=raw_data, 
                                                                                        params=params, 
                                                                                        print_details=False,
                                                                                        return_details=True)
print(sorted(ctd_list))
print(np.mean(ctd_list), np.median(ctd_list), sum(ctd_list))                                                                                    

import plotly.graph_objects as go


for i, x in enumerate(control_details):
    
    control_time, control_people_in, algo_people_in, ctd_term, room_id = x
                              
    algo_data = pl_count_list[i]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=algo_data["datetime"], 
            y=algo_data["CC"],
            mode="lines",
            name="Raw Occupancy"))
    
    fig.add_trace(
        go.Scatter(
            x=algo_data["datetime"],
            y=algo_data["CC_estimates"],
            mode="lines",
            name="Estimated Occupancy"))
    
    fig.update_layout(
        title=f"Room:{room_id} | Control Time:{control_time} <br><sup> Control Data: {control_people_in}, CTD: {ctd_term} </sup>",
        )
    # add vertical lines for control data
    fig.add_vline(x=control_time, line_width=1, line_dash="dash", line_color="black")
    fig.write_html(f"_plotting/plots/preprocessing_results/ctd-{ctd_term}_{room_id}_{control_time}.html")
    #fig.show()

########## Read out Results ########

#directory_filter = "discard"

#file_path = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.join(file_path, "_preprocessing", "results")
##parent_dir = parent_dir.split(" ")[0]
##directories = [x for x in list(os.walk(parent_dir))[0][1] if "with_5_6" in x]
#directories = [x for x in list(os.walk(parent_dir))[0][1] if directory_filter in x]

#ctd_list = []
#parameters_list = []
#for directory in directories:
#    files = list(os.walk(os.path.join(parent_dir, directory)))[0][2]
#    for i, file in enumerate(files):
#        with open(os.path.join(parent_dir, directory, file), "r") as file:
#            results = json.load(file)
            
#            parameters_list.append(results["parameters"])
#            ctd_list.append(results["CTD"])
            
#dataframe = pd.DataFrame({"parameters":parameters_list, "ctd":ctd_list})
#dataframe["mctd"] = dataframe["ctd"].apply(lambda x: np.mean(x))

#dict_list = []
##for  sort_by in sort_by_list:
#sort_by = ["mctd"]
#dataframe_sorted = dataframe.sort_values(by=sort_by)
#parameter_series_list = []
#for i,row in iter(dataframe_sorted[:10].iterrows()):
#    parameter_series = pd.json_normalize(row["parameters"], sep="-")
#    parameter_series_list.append(parameter_series)
#    #print(f"######## Combination: {i} ########")
#    #print(row["mse"], row["mae"], row["mctd"])
    
#parameters_df = pd.concat(parameter_series_list, axis=0)
#unique_values = [list(parameters_df[col].unique()) for col in parameters_df.columns]

#dict_uniqe_params = dict(list(zip(parameters_df.columns, unique_values)))
#print(dict_uniqe_params)

#results_dict = dict()
#for key in parameters_df.keys():
#    vcs = parameters_df[key].value_counts()
#    results_dict[key] = list(vcs.items())
#    file_name = "-".join(sort_by)
#    with open(os.path.join(file_path, f"results_{file_name}.json"), "w") as file:
#        json.dump(results_dict, file, indent=4)


#######################################################################################


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
#    for i,row in iter(dataframe_sorted[:25].iterrows()):
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

