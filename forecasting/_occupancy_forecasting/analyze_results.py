
from _occupancy_forecasting.data import load_data
import pandas as pd
import os
import ast
import numpy as np
import json

class StatsLogger():
    
    def __init__(self):
        self.comb_lists = []
        self.model_losses = []
        self.zero_baselines = []
        self.naive_baselines = []
        self.avg_baselines = []
        self.dataset_types = []
        self.loss_types = []
     
    def reset(self):
        self.comb_lists = []
        self.model_losses = []
        self.zero_baselines = []
        self.naive_baselines = []
        self.avg_baselines = []
        self.dataset_types = []
        self.loss_types = []
        
    def handle_array_types(self, array_type, array, dataset, loss_type):
        
        if array_type == "Combinations":
                        
            self.dataset_types.extend(np.repeat(dataset, len(array)))
            self.loss_types.extend(np.repeat(loss_type, len(array)))
            
            self.comb_lists.extend(array)
            
        elif array_type == "Model Losses":
            self.model_losses.extend(array)
            
        elif array_type == "BL zero Losses":
            self.zero_baselines.extend(array)
            
        elif array_type == "BL naive Losses":
            self.naive_baselines.extend(array)

        elif array_type == "BL avg Losses":
            self.avg_baselines.extend(array)

        else:
            print(array_type, array)
            raise ValueError("array_type not recognized")
            
    def return_dataframe(self, run_id):
        dataframe_dict = {
            "run_id": run_id,
            "dataset": self.dataset_types,
            "loss_type": self.loss_types,
            "combinations": self.comb_lists,
            "model_losses": self.model_losses,
        }
        if len(self.zero_baselines) > 0:
            dataframe_dict["zero_baselines"] = self.zero_baselines
        
        if len(self.naive_baselines) > 0:
            dataframe_dict["naive_baselines"] = self.naive_baselines
            
        if len(self.avg_baselines) > 0:
            dataframe_dict["avg_baselines"] = self.avg_baselines
        
        return pd.DataFrame(dataframe_dict)
    
    
class ResultsAnalyis:
    
    def __init__(self, path_to_resultsfile, path_to_data, path_to_checkpoints):
        
        self.path_to_resultsfile = path_to_resultsfile
        self.path_to_data = path_to_data
        self.path_to_checkpoints = path_to_checkpoints
        self.helper_path = os.path.join(path_to_data, "helpers")
        
        self.logger = StatsLogger()
        
        self.fundamental_features = ["occrate", "exam", "tutorium_test_cancelled", "registered", "type", "studyarea", "coursenumber", "dow", "hod", "weather", "avgocc"]
    
    ####### Loading results and parsing them #######
    def load_results(self):
        
        with open(self.path_to_resultsfile, "r") as f:
            lines = f.readlines()
            line_str = "".join(lines)
            
        list_of_runs = line_str.split("\n\n\n")
        list_of_runs = [run for run in list_of_runs if run != ""][:-1]
        
        return list_of_runs
    
    def parse_run(self, run_corrected):
        
        for elem in run_corrected:
            
            by_bar = elem.split("|")

            if len(by_bar) == 2:
                dataset = by_bar[0].split(":")[1].strip()
                loss_type = by_bar[1].split(":")[1].strip()

            elif len(by_bar) == 1:
                
                if by_bar[0] == "":
                    continue
                
                array_type, array = by_bar[0].split(":")
                array_type = array_type.strip()
                
                if array_type == "Hyperparameters":
                    continue
                
                array = array.strip()
                array = np.array(ast.literal_eval(array))
                self.logger.handle_array_types(array_type, array, dataset, loss_type)
                    
    def parse_list_of_runs(self, list_of_runs):
        
        list_of_dfs = []
        for run in list_of_runs:
            
            splitted_run = run.split("\n")
            # filter out empty strings
            splitted_run = [elem for elem in splitted_run if elem != ""]

            # correct split such that \n does not happen inside and elements
            splitted_run_corrected = []
            for elem in splitted_run[2:]:
                
                by_bar = elem.split("|")
                if len(by_bar) == 2:
                    splitted_run_corrected.append(elem)
                    
                elif len(by_bar) == 1:
                    if by_bar[0] == "":
                        continue

                    if len(by_bar[0].split(":")) == 2:
                        splitted_run_corrected.append(elem)
                    else:
                        splitted_run_corrected[-1] = splitted_run_corrected[-1] + elem
                        
                else:
                    raise
            
            run_id = int(splitted_run[0].split(" ")[2])
            #if (run_id < 13) or (run_id > 16):
            #    continue
            self.logger.reset()
            
            self.parse_run(splitted_run_corrected)
            
            run_df = self.logger.return_dataframe(run_id)
            
            list_of_dfs.append(run_df)
            
        df_results = pd.concat(list_of_dfs).reset_index(drop=True)
        
        return df_results
    
    def add_hyperparameters(self, parsed_results):
        
        dataframe = parsed_results.copy(deep=True)
        for idx, row in dataframe.iterrows():
            
            comb = row["combinations"]
    
            comb_path = os.path.join(self.path_to_checkpoints, f"run_{comb[0]}/comb_{comb[1]}")
            hyperparameters_path = os.path.join(comb_path, "hyperparameters.json")

            hyperparameters = json.load(open(hyperparameters_path, "r"))
            
            # overwrite combinations with tuple of run_id and comb_id
            dataframe.at[idx, "combinations"] = (row["run_id"], comb[1])
            
            # add all hyperparameters to the dataframe
            for key, value in hyperparameters.items():
                dataframe.at[idx, key] = str(value)
        
        return dataframe
    
    def data_preparation(self):
        
        list_of_runs = self.load_results()
        parsed_results = self.parse_list_of_runs(list_of_runs)
        parsed_results = self.add_hyperparameters(parsed_results)
        
        return parsed_results
    
    def filter_clean_pivot(self, dataframe, filter_dict):
        
        filtered_dataframe = self.filter_dataframe_by_dict(dataframe, filter_dict)
        
        cleaned_dataframe = filtered_dataframe[["dataset", "features", "run_id", "model_losses"]]
        
        pivot_dataframe = cleaned_dataframe.pivot(index=['features', 'run_id'], columns='dataset', values='model_losses').reset_index()
        pivot_dataframe.columns = ['features', 'run_id', 'test_loss', 'val_loss']
        
        return pivot_dataframe
    
    ###### Filter functions ######
    def filter_dataframe_by_column_value(self, dataframe, column, value):
        return dataframe[dataframe[column] == value].reset_index(drop=True)
    
    def filter_dataframe_by_dict(self, dataframe, filter_dict):
        for key, value in filter_dict.items():
            dataframe = self.filter_dataframe_by_column_value(dataframe, key, value)
        return dataframe
    
    ##### Grouping functions ######
    def group_by_features(self, pivot_dataframe):
        
        return pivot_dataframe.groupby('features').agg(
            mean_validation_mae=('val_loss', 'mean'),
            std_validation_mae=('val_loss', 'std'),
            mean_test_mae=('test_loss', 'mean'),
            std_test_mae=('test_loss', 'std')
        ).reset_index()
        
    def group_by_subfeatures(self, pivot_dataframe):
        
        dataframe = pivot_dataframe.copy(deep=True)
        
        all_features = set(list(dataframe['features']))
        for feature in all_features:
            dataframe[feature] = dataframe['features'].apply(lambda x: feature in x)

        # Group by individual features and calculate mean/std
        feature_analysis = {}
        for feature in all_features:
            feature_analysis[feature] = {
                'mean_validation_mae': dataframe.loc[dataframe[feature], 'val_loss'].mean(),
                'std_validation_mae': dataframe.loc[dataframe[feature], 'val_loss'].std(),
                'mean_test_mae': dataframe.loc[dataframe[feature], 'test_loss'].mean(),
                'std_test_mae': dataframe.loc[dataframe[feature], 'test_loss'].std()
            }

        return_df = pd.DataFrame.from_dict(feature_analysis, orient='index').sort_index().reset_index()
        return return_df
    
    def group_by_fundamental_features(self, pivot_dataframe):
        
        dataframe = pivot_dataframe.copy(deep=True)
        
        all_features = self.fundamental_features
        for feature in all_features:
            dataframe[feature] = dataframe['features'].apply(lambda x: feature in x)

        # Group by individual features and calculate mean/std
        feature_analysis = {}
        for feature in all_features:
            feature_analysis[feature] = {
                'mean_validation_mae': dataframe.loc[dataframe[feature], 'val_loss'].mean(),
                'std_validation_mae': dataframe.loc[dataframe[feature], 'val_loss'].std(),
                'mean_test_mae': dataframe.loc[dataframe[feature], 'test_loss'].mean(),
                'std_test_mae': dataframe.loc[dataframe[feature], 'test_loss'].std()
            }

        return_df = pd.DataFrame.from_dict(feature_analysis, orient='index').sort_index().reset_index()
        return return_df
    