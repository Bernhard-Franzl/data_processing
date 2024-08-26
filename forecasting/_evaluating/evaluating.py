# Imports for ParameterSearch
import itertools
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Imports for Evaluator
import pandas as pd
from pandas.tseries.offsets import DateOffset

from _preprocessing import SignalAnalyzer
from _preprocessing import PLCount


room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}

def write_results_to_txt(file_name, comb_number, params, ctd_list):
    with open(file_name, "a") as file:
        file.write(f"######## Combination: {comb_number} ########\n")
        file.write(f"Parameters: {params}\n")
        #file.write(f"MSE: {np.round(np.mean(se_list), 4)} MedianSE: {np.median(se_list)}\n")
        #file.write(f"MAE: {np.round(np.mean(ae_list), 4)} MedianAE: {np.median(ae_list)}\n")
        file.write(f"MCTD: {np.round(np.mean(ctd_list), 4)} MedianCTD: {np.median(ctd_list)}\n")
        file.write("\n")
        file.write("\n")
        
    return None

def write_results_to_json(file_name, params, ctd_list):
    
    with open(file_name, "w") as file:
        json.dump({"parameters":params, 
                   #"SE":se_list, 
                   #"AE":ae_list, 
                   "CTD":ctd_list}, file)
    
        
    return None

class ParameterSearch:
    
    def __init__(self, path_to_json=None, parameter_dict=None):
        
        # check if either path_to_parameters or parameter_dict is given
        if path_to_json is None and parameter_dict is None:
            raise ValueError("Either path_to_parameters or parameter_dict must be given")
        # check if both are given
        elif path_to_json is not None and parameter_dict is not None:
            raise ValueError("Either path_to_parameters or parameter_dict must be given")
        # check if path_to_json is given
        elif path_to_json is not None:
            # read json file
            with open(path_to_json, "r") as file:
                self.parameter_dict = json.load(file)
        # check if parameter_dict is given
        elif parameter_dict is not None:
            self.parameter_dict = parameter_dict
        else:
            raise ValueError("Something went wrong!")
          
    #### Generate all possible combinations of parameters ####
    def _extract_all_combinations(self, params_dict):
    
        values = []
        keys = []
        for key in params_dict.keys():
            
            sub_dict = params_dict[key]
            
            for sub_key in sub_dict.keys():
                values.append(sub_dict[sub_key])
                keys.append((key, sub_key))
        
        n_combinations = np.product([len(value) for value in values])

        return keys, iter(itertools.product(*values)), n_combinations
     
    def _generate_comb_dict(self, keys, combination):
        dictionary = defaultdict(dict)
        for key, value in zip(keys, combination):
                dictionary[key[0]][key[1]] = value
        return dictionary
        
    def _combinations_wrapper(self, params_dict, tqdm_bar):
        
        keys, comb_iterator, n_combs = self._extract_all_combinations(params_dict=params_dict)
        
        if tqdm_bar:
            tqdm_iterator = tqdm(comb_iterator, total=n_combs)
            for x in tqdm_iterator:
                yield self._generate_comb_dict(keys, x)
        else:
            for x in comb_iterator:
                yield self._generate_comb_dict(keys, x)
    
    def combinations_iterator(self, tqdm_bar=True):
        return self._combinations_wrapper(params_dict=self.parameter_dict, tqdm_bar=tqdm_bar)

    #### Simple Grid Search ####
    def _extract_all_combinations_simple(self, params_dict):
        
        values = []
        keys = []
        for key in params_dict.keys():
            values.append(params_dict[key])
            keys.append(key)
        
        n_combinations = np.prod([len(value) for value in values])

        return keys, iter(itertools.product(*values)), n_combinations
    
    def _grid_search_wrapper(self, params_dict, tqdm_bar):
        
        keys, comb_iterator, n_combs = self._extract_all_combinations_simple(params_dict=params_dict)
        
        if tqdm_bar:
            tqdm_iterator = tqdm(comb_iterator, total=n_combs)
            for x in tqdm_iterator:
                yield dict(zip(keys, x))
        else:
            for x in comb_iterator:
                yield dict(zip(keys, x))
                
    def grid_search_iterator(self, tqdm_bar=True):
        return self._grid_search_wrapper(params_dict=self.parameter_dict, tqdm_bar=tqdm_bar)
      
class Evaluator:
    
    # Helper class for evaluating the functions of the SignalAnalyzer class
    def __init__(self, class_name, path_to_control_data):
        
        self.control_data = pd.read_csv(path_to_control_data)
        self.control_data["room_id"] = self.control_data["room"].map(room_to_id)
        
        self.class_name = class_name
        if self.class_name == "SignalAnalyzer":
            self.class_to_evaluate = SignalAnalyzer()
            self.function_to_evaluate = self.class_to_evaluate.calc_participants
            
        elif self.class_name == "PLCount":
            self.class_to_evaluate = PLCount()
            self.function_to_evaluate = self.class_to_evaluate.run_algorithm_vectorized
            
        else:
            raise ValueError("Class not recognized")
        
    ######################## SignalAnalyzer ########################
    def prepare_control_data_signal_analyzer(self, row):
        
        start_time_int = row["start_time"]
        end_time_int = row["end_time"]
        time_string = row["time"].split(":")
        date = row["date"].split(".")
        day = int(date[0])
        month = int(date[1])
        
        control_time = datetime(2024, month, day, int(time_string[0]), int(time_string[1]), 0)
        start_time = datetime(2024, month, day, start_time_int//100, start_time_int%100, 0)
        end_time = datetime(2024, month, day, end_time_int//100, end_time_int%100, 0)
        control_people_in = row["people_in"]
        room_id = row["room_id"]
        first = bool(row["first"])
        last = bool(row["last"])
        
        return control_time, start_time, end_time, control_people_in, room_id, first, last
    
    def prepare_real_data_signal_analyzer(self, dataframe:pd.DataFrame, room_id:int):
        df = dataframe.copy()
        df = self.class_to_evaluate.filter_by_room(df, room_id)
        return df
    
    def plot_function_signal_analyzer(self, data:pd.DataFrame, title, file_name, control_time):
        
        
        #horizontal_lines = [(participants[0], "black", " before"),
        #                    (participants[1], "gold", " after")]

        vertical_lines = [(control_time, "green", "control_time"),]
    
  
        self.class_to_evaluate.plot_participants_algo(file_name = file_name,
                                participants=data,
                                df_list = None,
                                control = None,
                                extrema = None,
                                horizontal_lines=[],
                                vertical_lines=vertical_lines,
                                title=title)
    
    def get_prediction(self, participants, mode):
        if mode == "mean":
            return int(np.mean(participants))
        elif mode == "max":
            return int(np.max(participants))
        elif mode == "min":
            return int(np.min(participants))
        elif mode == "first":
            return int(participants[0])
        elif mode == "second":
            return int(participants[1])
        else:
            raise ValueError("Mode not recognized")
        
    def evaluate_signal_analyzer(self, data:pd.DataFrame, params:dict, raw_data=None, details=False):
        
        signal_params = params["signal_params"]
        
        ae_list = []
        se_list = []
        ctd_list = []
            
        for index, row in self.control_data.iterrows():
            
            control_time, start_time, end_time, control_people_in, room_id, first, last = self.prepare_control_data_signal_analyzer(row)
            
            df_real = self.prepare_real_data_signal_analyzer(data, row["room_id"])

            # m is an extremely important parameter -> the one that is used to calculate the extrema
            df_list, participants, extrema, df_list_plotting, control = self.function_to_evaluate(
                                                    dataframe=df_real, 
                                                    control=True,
                                                    start_time=start_time,
                                                    end_time=end_time,
                                                    first=first,
                                                    last=last,
                                                    params=signal_params)


            df_plotting = self.class_to_evaluate.merge_participant_dfs(df_list_plotting)
            
            # plotting functions somehow messes with data reading -> investigate!!!
            #self.plot_function_signal_analyzer(data=df_plotting, 
            #                                   title=f"Control:{control_people_in}, Time:{control_time}", 
            #                                   file_name=f"{control_time}_{room_id}.png", 
            #                                   control_time=control_time)
            
            control_row = df_plotting[df_plotting["time"] == control_time]
            
            #prediction = self.get_prediction(participants, signal_params["prediction_mode"])
            
            diff_ratio = abs(np.diff(participants))/max(participants)
            
            if diff_ratio > signal_params["max_mean_cutoff"]:
                prediction = int(np.max(participants))
            else:
                prediction = int(np.mean(participants))
            

            mse_term = (control_people_in - prediction)**2
            ae_term = abs(control_people_in - prediction)
            
            ctd_term = abs(control_people_in - int(control_row["people_inside"].values[0]))
            
            if details:
                print(f"###### Resutls Index:{index} ######")
                print(f"Room: {room_id}, Start Time: {start_time}, End Time: {end_time}")
                print(f"------------------------------------")
                print("Control Time Real:",control_people_in)
                print("Control Time Algo:", int(control_row["people_inside"].values[0]))
                print("CTD:", ctd_term)
                print(f"------------------------------------")
                print("Participants:", participants)
                print("Prediction:", prediction, "Max_mean_cutoff", signal_params["max_mean_cutoff"])
                print("SE:", mse_term, "AE:", ae_term)
                
                #ae_list_helper = []
                #se_list_helper = []
                ##mode_list = ["max", "mean", "min"]
                ##for mode in mode_list:
                    
                ##    prediction = self.get_prediction(participants, mode)
                ##    se = (control_people_in - prediction)**2
                ##    ae = abs(control_people_in - prediction)
                    
                ##    print(f"Mode: {mode}, Pred: {prediction}")
                ##    print(f"SE: ", se, "AE: ", ae)
                    
                ##    ae_list_helper.append(ae)
                ##    se_list_helper.append(se)
                
                #se_arg_min = np.argmin(se_list_helper)
                #ae_arg_min = np.argmin(ae_list_helper)
                #se_min_tuple = (mode_list[se_arg_min], se_list_helper[se_arg_min])
                #ae_min_tuple = (mode_list[ae_arg_min], ae_list_helper[ae_arg_min], control_people_in, int(control_row["people_inside"].values[0]), participants, df_plotting)
                
                #se_min_list.append(se_min_tuple)
                #ae_min_list.append(ae_min_tuple)
                
                print(f"------------------------------------")
                print()
                
                #raw_to_save = self.class_to_evaluate.filter_by_room(raw_data, room_id)
                #raw_to_save = self.class_to_evaluate.filter_by_time(raw_to_save,
                #                                      start_time-timedelta(minutes=15),
                #                                      end_time+timedelta(minutes=15))
                #raw_to_save.sort_values(by="time", inplace=True)
                #raw_to_save.to_csv(f"data/data_index:{index}_{room_id}_{control_time}_.csv")
            
            ae_list.append(ae_term)
            se_list.append(mse_term)
            ctd_list.append(ctd_term)
            
        return se_list, ae_list, ctd_list     
    
    ######################## PLCount ########################
    def prepare_control_data_plcount(self, row, day_timestamp):
        
        time_string = row["time"].split(":")
        control_time = datetime(2024, day_timestamp.month, day_timestamp.day, int(time_string[0]), int(time_string[1]), 0)
        
        control_people_in = row["people_in"]
        
        return control_time, control_people_in
    
    def process_grouping(self, grouping):
        
        date = grouping[0].split(".")
        day = int(date[0])
        month = int(date[1])
        day_timestamp = datetime(2024, month, day, 0, 0, 0)
        
        room_id = grouping[1]
        
        return day_timestamp, room_id
        
    def evaluate_pl_count(self, data:pd.DataFrame, params:dict, dfguru, raw_data=None, details=False):
        
        plcount_params = params["plcount_params"]
        
        occupancy_count_list = []
        ctd_list = []
        for grouping, control_subdf in self.control_data.groupby(["date", "room_id"]):

            day_timestamp, room_id = self.process_grouping(grouping)
            
            # filter by time
            df_day = dfguru.filter_by_timestamp(data, "datetime",
                                        day_timestamp, day_timestamp + DateOffset(days=1))
            # filter by room_id
            df_day_room = dfguru.filter_by_roomid(df_day, room_id)

            occupancy_counts = dfguru.calc_occupancy_count(df_day_room, "datetime", "1min")
            occupancy_counts["delta_CC"] = self.class_to_evaluate.calc_delta(occupancy_counts, "CC")
            occupancy_counts["sigma"] = self.class_to_evaluate.calc_sigma(occupancy_counts, "delta_CC")
        
            cc_max = occupancy_counts.CC.max()
            m = int(cc_max + (cc_max*plcount_params["cc_max_factor"]))
            n = len(occupancy_counts.datetime)
            
            estimates = self.function_to_evaluate(n, m, occupancy_counts["delta_CC"], occupancy_counts["sigma"])
            occupancy_counts["CC_estimates"] = estimates
            occupancy_counts = occupancy_counts.drop(columns=["delta_CC", "sigma"])      
                              
            for index, row in control_subdf.iterrows():
            
                control_time, control_people_in = self.prepare_control_data_plcount(row, day_timestamp)

                CC_row = occupancy_counts[occupancy_counts["datetime"] == control_time]
                algo_people_in = int(CC_row["CC_estimates"].values[0])
                ctd_term = abs(control_people_in - algo_people_in)
                
                ctd_list.append(ctd_term)
                
                if ctd_term > 1:
                    if details:
                        print(f"###### Resutls Index:{index} ######")
                        print(f"Room: {room_id}, Time: {control_time}")
                        print(f"------------------------------------")
                        print("Control Time Real:",control_people_in)
                        print("Control Time Algo:", algo_people_in)
                        print("CTD:", ctd_term)
                        print(f"------------------------------------")

        return ctd_list

            
        return None