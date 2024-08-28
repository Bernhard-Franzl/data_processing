# import DateOffset
from pandas.tseries.offsets import DateOffset
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

rng = np.random.default_rng(seed=42)

def train_val_test_split(data_dict, verbose=True):
    
    # randomly exclude chunks of the data
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    total_size = 0
    val_size = 0
    test_size = 0
    train_size = 0
    
    for room_id in data_dict:

        occ_time_series = data_dict[room_id]
        total_size += len(occ_time_series)
        
        # generate chunks with size 0.05 of the data
        index_shift = int(len(occ_time_series) * 0.05)
        indices = np.arange(0, len(occ_time_series), index_shift)
        if (len(occ_time_series)-indices[-1]) < index_shift:
            indices = indices[:-1]
            
        rng.shuffle(indices)
        
        val_indices = indices[:2]
        test_indices = indices[2 : 4]
        train_indices = indices[4:]
        
        ts_val = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in val_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
        ts_test = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in test_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
        ts_train = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in train_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
        
        val_size += len(ts_val)
        test_size += len(ts_test)
        train_size += len(ts_train)
        
        train_dict[room_id] = ts_train
        val_dict[room_id] = ts_val
        test_dict[room_id] = ts_test
    
    if verbose:
        print("############## Split Summary ##############")
        print("Size of Validationset:", val_size/total_size)
        print("Size of Testset:", test_size/total_size)
        print("Size of Trainset:", train_size/total_size)
        print()
    
    return train_dict, val_dict, test_dict
        
class OccFeatureEngineer():
    
    course_features = {"exam", "lecture"}
    datetime_features = {"weekday"}
    permissible_features = course_features.union(datetime_features)
        
    def __init__(self, cleaned_occ_data, course_dates_data, course_info_data, dfguru):

        self.occ_time_series = cleaned_occ_data
        min_timestamp = self.occ_time_series["datetime"].min().replace(hour=0, minute=0, second=0, microsecond=0)
        max_timestamp = self.occ_time_series["datetime"].max().replace(hour=0, minute=0, second=0, microsecond=0) + DateOffset(days=1)
        
        self.dfg = dfguru
        
        self.course_dates_table = dfguru.filter_by_timestamp(
            dataframe = course_dates_data,
            start_time = min_timestamp,
            end_time = max_timestamp,
            time_column = "start_time"
        )

        self.course_info_table = course_info_data
         
    def derive_features(self, features, room_id=None):
        
        # get the course number
        occ_time_series = self.occ_time_series.copy(deep=True)

        # check if features are permissible
        feature_set = set(features)
        set_diff = feature_set.difference(self.permissible_features)
        if set_diff:
            raise ValueError(f"Features {set_diff} are not permissible.")
        
        # check if features are already present
        feature_set = feature_set.difference(occ_time_series.columns)

        # add course features
        course_features = self.course_features.intersection(feature_set)
        if course_features:
            occ_time_series = self.add_course_features(occ_time_series, course_features, room_id)
        
        # add datetime features
        datetime_features = self.datetime_features.intersection(feature_set)
        if datetime_features:
            occ_time_series = self.add_datetime_features(occ_time_series, datetime_features)

        return occ_time_series
    
    def add_course_features(self, time_series, features, room_id):

        # room_id must be not None
        if room_id is None:
            raise ValueError("Room ID must be provided.")
        
        course_dates_in_room = self.dfg.filter_by_roomid(self.course_dates_table, room_id)
        
        # initialize all features to 0
        time_series["course_number"] = 0
        for feature in features:
            time_series[feature] = 0
            
        for grouping, sub_df in course_dates_in_room.groupby(["start_time", "end_time"]):
            
            # get course time_span
            course_time_mask = (time_series["datetime"] >= grouping[0]) & (time_series["datetime"] <= grouping[1])
            
            # get course_number
            time_series.loc[course_time_mask, "course_number"] = ",".join(sub_df["course_number"].values)
            
            # get course features
            if "exam" in features:
                time_series.loc[course_time_mask, "exam"] = int(sub_df["exam"].values[0])
                
            if "lecture" in features:
                time_series.loc[course_time_mask, "lecture"] = 1
        
        return time_series

class OccupancyDataset(Dataset):
    
    def __init__(self, time_series_dict: dict, frequency:str, x_size:int=24, y_size:int=24, verbose:bool=True):
        """ Constructor for the occupancy dataset
        Task: Convert the cleaned data into a list of samples
        """
        super().__init__()
        
        # the time series must be structured as a dictionary with the room_id as the key
        self.time_series_dict = time_series_dict
        self.room_ids = list(time_series_dict.keys())
        
        # convert frequency to timedelta
        td_freq = pd.to_timedelta(frequency)
        
        # window size is 1 day
        self.x_size = x_size
        self.y_size = y_size
        
        self.verbose = verbose
        

        #print("############## Dataset Summary ##############")
            
        self.samples = []
        for room_id in self.room_ids:
            
            #print("Sample Generation for Room ID: ", room_id)
            occ_time_series = self.time_series_dict[room_id]
            
            # check for holes in the time series -> they break the rolling window
            ts_diff = occ_time_series["datetime"].diff()
            holes = occ_time_series[ts_diff > td_freq]
            
            if holes.empty:
                #print("Check 1: No holes found in the time series.")
                info, X, y = self.create_samples(occ_time_series, room_id)
                self.samples.extend(list(zip(info, X, y)))
                #print( "-----------------------------------")
                
            else:
                #print("Check 1: Holes found and taken care of.")
                
                cur_idx = 0
                for hole_idx in holes.index:
                    info, X, y = self.create_samples(occ_time_series.iloc[cur_idx:hole_idx], room_id)
                    self.samples.extend(list(zip(info, X, y)))
                    cur_idx = hole_idx
                
                # add the last part of the time series
                info, X, y = self.create_samples(occ_time_series.iloc[cur_idx:], room_id)
                self.samples.extend(list(zip(info, X, y)))
                #print( "-----------------------------------")
        #print()
                    
    def create_samples(self, time_series, room_id):
        
        occ_time_series = time_series.copy(deep=True)
        occ_time_series["day"] = occ_time_series["datetime"].dt.date
        
        X_list = []
        y_list = []
        sample_info = []
    
        # we want to predict the next N steps based on the previous T steps
        window_size = self.x_size + self.y_size
        for window in occ_time_series.rolling(window=window_size):
            # skip the first window_size elements only consider full windows
            
            X_df = window.iloc[:self.x_size]
            X = torch.Tensor(X_df["CC_estimates"].values)
            
            y_df = window.iloc[self.x_size:]
            y = torch.Tensor(y_df["CC_estimates"].values)

            X_list.append(X)
            y_list.append(y)
            
            sample_info.append((room_id, X_df["datetime"], y_df["datetime"]))

        sanity_check_1 = [len(x)==self.x_size for x in X_list[window_size-1:]]
        sanity_check_2 = [len(y)==self.y_size for y in y_list[window_size-1:]]
        
        if (all(sanity_check_1) & all(sanity_check_2)):
            #print("Check 2: All the samples have the correct size.")
            return sample_info[window_size-1:], X_list[window_size-1:], y_list[window_size-1:]
        
        else:
            raise ValueError("Sanity Check Failed")
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
