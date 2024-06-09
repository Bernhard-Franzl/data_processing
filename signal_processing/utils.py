# Imports for ParameterSearch
import itertools
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm

# Imports for Evaluator
import pandas as pd
from signal_analysis import SignalAnalyzer
from datetime import datetime

room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}

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
    
    
class Evaluator:
    
    # Helper class for evaluating the functions of the SignalAnalyzer class
    def __init__(self, class_name, path_to_control_data):
        
        self.control_data = pd.read_csv(path_to_control_data)
        self.control_data["room_id"] = self.control_data["room"].map(room_to_id)
        
        self.class_name = "SignalAnalyzer"
        if self.class_name == "SignalAnalyzer":
            self.class_to_evaluate = SignalAnalyzer()
            self.function_to_evaluate = self.class_to_evaluate.calc_participants
    
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
        
    def evaluate_signal_analyzer(self, data:pd.DataFrame, params:dict):
        
        mse = 0
        ae = 0
        for index, row in self.control_data.iterrows():
            
            control_time, start_time, end_time, control_people_in, room_id, first, last = self.prepare_control_data_signal_analyzer(row)
            
            df_real = self.prepare_real_data_signal_analyzer(data, row["room_id"])
        
            # m is an extremely important parameter -> the one that is used to calculate the extrema
            df_list, participants, extrema, df_list_plotting, control = self.function_to_evaluate(
                                                    dataframe = df_real, 
                                                    control=True,
                                                    mode="median",
                                                    start_time=start_time,
                                                    end_time=end_time,
                                                    first=first,
                                                    last=last)
        
            df_plotting = self.class_to_evaluate.merge_participant_dfs(df_list_plotting)
            
            # plotting functions somehow messes with data reading -> investigate!!!
            #self.plot_function_signal_analyzer(data=df_plotting, 
            #                                   title=f"Control:{control_people_in}, Time:{control_time}", 
            #                                   file_name=f"{control_time}_{room_id}.png", 
            #                                   control_time=control_time)
            
            control_row = df_plotting[df_plotting["time"] == control_time]
            prediction = int(np.max(participants))

            mse_term = (control_people_in - prediction)**2
            mse += mse_term
            ae_term = abs(control_people_in - prediction)
            ae += ae_term
        
            #if mse_term > 10:
            #    print("##################")
            #    print("Time: ", control_time)
            #    print("Room: ", room_id)
            #    print("Participants: ", participants)
            #    print("Control: ", control_people_in)
            #    print("Prediction: ", prediction)
            #    print("MSE: ", mse_term)
            #    print("AE: ", ae_term)
            #    print("##################")
            #    print()   
        
        return mse/len(self.control_data), ae/len(self.control_data)         
                             
            #prediction = control_row["people_inside"].values[0]
            #Mode: Mean 
            #MSE:  88.55172413793103
            #AE:  4.0
            #Mode: Median
            #MSE: 88.55172413793103
            #AE: 4.0
        
            #prediction = int(np.mean(participants))
            # Mode: Mean
            #MSE:  32.10344827586207
            #AE:  3.68965517241379
            #Mode: Median
            #MSE: 31.689655172413794
            #AE:  3.6206896551724137
        
            #prediction = int(np.max(participants))
            # Mode: Mean
            #MSE:  32.10344827586207
            #AE:  3.68965517241379
            #Mode: Median
            #MSE:  11.96551724137931
            #AE:  2.793103448275862
        
            #prediction = int(np.min(participants))
            #Mode: Mean, filter
            #MSE:  100.89655172413794
            #AE:  5.0344827586206895
            #Mode: Median
            #MSE:  105.10344827586206
            #AE:  5.24137931034482
        
            # try first of participants
            #prediction = participants[0]
            # try second of participants
            #prediction = participants[1]