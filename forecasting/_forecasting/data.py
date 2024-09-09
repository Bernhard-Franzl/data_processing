# import DateOffset
from pandas.tseries.offsets import DateOffset
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def load_data_dicts(path_to_data_dir, frequency, dfguru):
    
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    for room_id in [0, 1]:
        for type in ["train", "test", "val"]:
            
            df = dfguru.load_dataframe(
                path_repo=path_to_data_dir, 
                file_name=f"room-{room_id}_freq-{frequency}_{type}_dict")
            
            if type == "train":
                train_dict[room_id] = df
            elif type == "val":
                val_dict[room_id] = df
            elif type == "test":
                test_dict[room_id] = df
            else:
                raise ValueError("Unknown type.")
            
    return train_dict, val_dict, test_dict
    
def prepare_data(path_to_data_dir, frequency, feature_list, dfguru, rng):
    
    course_dates_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_dates")
    
    course_info_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_info")
    
    data_dict = {}
    for room_id in [0, 1]:
        
        ########## Load Data ##########
        occ_time_series = dfguru.load_dataframe(
            path_repo=path_to_data_dir, 
            file_name=f"room-{room_id}_freq-{frequency}_cleaned_data_29_08", 
        )
        
        ########## OccFeatureEngineer ##########

        occ_time_series = OccFeatureEngineer(
            occ_time_series, 
            course_dates_data, 
            course_info_data, 
            dfguru,
            frequency
        ).derive_features(
            features=feature_list, 
            room_id=room_id
        )
          
        data_dict[room_id] = occ_time_series
        
    train_dict, val_dict, test_dict = train_val_test_split(data_dict, rng, verbose=True)
    
    return train_dict, val_dict, test_dict
    
def train_val_test_split(data_dict, rng, verbose=True):
    
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
        
        test_slice = int(len(indices) * 0.15)
        val_indices = indices[:test_slice]
        test_indices = indices[test_slice : 2*test_slice]
        train_indices = indices[2*test_slice:]
        
        ts_val = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in val_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
        ts_test = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in test_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
        ts_train = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in train_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
        
        val_size += len(ts_val)
        test_size += len(ts_test)
        train_size += len(ts_train)
        
        train_dict[room_id] = ts_train
        val_dict[room_id] = ts_val
        test_dict[room_id] = ts_test
        
        print(room_id, sorted(train_indices))
        
    if verbose:
        print("############## Split Summary ##############")
        print("Size of Validationset:", val_size/total_size)
        print("Size of Testset:", test_size/total_size)
        print("Size of Trainset:", train_size/total_size)
        print()
    
    return train_dict, val_dict, test_dict
        
        
class OccFeatureEngineer():
    
    course_features = {"exam", "lecture", "registered", "test", "tutorium", "type"}
    datetime_features = {"dow", "hod", "week"}
    general_features = {"occcount", "occrate"}
    shift_features = {"occcount1week", "occrate1week", "occcount1day", "occrate1day"}
    permissible_features = course_features.union(datetime_features)
    permissible_features = permissible_features.union(general_features)
    permissible_features = permissible_features.union(shift_features)
        
    def __init__(self, cleaned_occ_data, course_dates_data, course_info_data, dfguru, frequency):

        self.occ_time_series = cleaned_occ_data
        min_timestamp = self.occ_time_series["datetime"].min().replace(hour=0, minute=0, second=0, microsecond=0)
        max_timestamp = self.occ_time_series["datetime"].max().replace(hour=0, minute=0, second=0, microsecond=0) + DateOffset(days=1)
        
        self.dfg = dfguru
        self.frequency = frequency
        self.course_dates_table = dfguru.filter_by_timestamp(
            dataframe = course_dates_data,
            start_time = min_timestamp,
            end_time = max_timestamp,
            time_column = "start_time"
        )

        self.course_info_table = course_info_data  
        
        self.course_types = self.course_info_table["type"].unique()   
             
    def derive_features(self, features, room_id):
        
        # get the course number
        occ_time_series = self.occ_time_series.copy(deep=True)

        # check if features are permissible
        feature_set = set(features)
        set_diff = feature_set.difference(self.permissible_features)
        if set_diff:
            raise ValueError(f"Features {set_diff} are not permissible.")
        
        # check if features are already present
        feature_set = feature_set.difference(occ_time_series.columns)

        # Add general features 
        general_features = self.general_features.intersection(feature_set)
        occ_time_series = self.add_general_features(occ_time_series, general_features, room_id)
        
        # add course features
        course_features = self.course_features.intersection(feature_set)
        if course_features:
            occ_time_series = self.add_course_features(occ_time_series, course_features, room_id)
        
        # add datetime features
        datetime_features = self.datetime_features.intersection(feature_set)
        if datetime_features:
            occ_time_series = self.add_datetime_features(occ_time_series, datetime_features)

        # add shift features
        shift_features = self.shift_features.intersection(feature_set)
        if shift_features:
            occ_time_series = self.add_shift_features(occ_time_series, shift_features)
        
        return occ_time_series
    
    ########### General Features ############
    def add_shift_features(self, time_series, features):
        # initialize all features to 0
        for feature in features:
            time_series[feature] = 0
            
        if "occount1week" in features:
            
            time_series = self.shift_n_days(time_series, "occcount", "occcount1week", 7)
            time_series = self.shift_n_days(time_series, "occcountdiff", "occcountdiff1week", 7)  
            
        if "occcount1day" in features:
            
            time_series = self.shift_n_days(time_series, "occcount", "occcount1day", 1)
            time_series = self.shift_n_days(time_series, "occcountdiff", "occcountdiff1day", 1)            
        
        if "occrate1week" in features:
            
            time_series = self.shift_n_days(time_series, "occrate", "occrate1week", 7)
            time_series = self.shift_n_days(time_series, "occratediff", "occratediff1week", 7)
            
        if "occrate1day" in features:
            
            time_series = self.shift_n_days(time_series, "occrate", "occrate1day", 1)
            time_series = self.shift_n_days(time_series, "occratediff", "occratediff1day", 1)
        
        return time_series
    def shift_n_days(self, time_series, column_in, column_out, n_days):
        factor = pd.to_timedelta("1d") / pd.to_timedelta(self.frequency)
        time_series[column_out] = time_series[column_in].shift(int(n_days * factor))
        time_series[column_out] = time_series[column_out].fillna(-1)
        return time_series
        
    def add_general_features(self, time_series, features, room_id):
        
        # check if one of the general features is requested
        if not features:
            raise ValueError("At least one general feature must be requested.")
        
        # initialize all features to 0
        for feature in features:
            time_series[feature] = 0

        ################################################################################################
        if "occcount" in features:
            time_series["occcount"] = time_series["CC_estimates"]
            time_series["occcountdiff"] = time_series["occcount"].diff(1).combine_first(time_series["occcount"])

        ################################################################################################
        
        if "occrate" in features:
            if room_id is None:
                raise ValueError("For feature 'occrate' room ID must be provided.")
        
            room_capa = self.dfg.filter_by_roomid(self.course_dates_table, room_id)["room_capacity"].unique()
            time_series["occrate"] = time_series["CC_estimates"] / int(room_capa)
            time_series["occratediff"] = time_series["occrate"].diff(1).combine_first(time_series["occrate"])
            del room_capa
            # differencing
            #time_series["undiffed_occrate"] = time_series["occrate_diff"].cumsum()
            #print(time_series[["occrate","occrate_diff", "undiffed_occrate"]])
            # raise
            
        ################################################################################################
            
        time_series.drop(columns=["CC_estimates", "CC"], inplace=True)
        return time_series
    
    ########### Course Features ############
    def add_course_features(self, time_series, features, room_id):

        # room_id must be not None
        if room_id is None:
            raise ValueError("Room ID must be provided.")
        
        course_dates_in_room = self.dfg.filter_by_roomid(self.course_dates_table, room_id)
        
        # initialize all features to 0
        time_series["course_number"] = ''
        for feature in features:
            time_series[feature] = 0
            
        if "type" in features:
            time_series.drop(columns=["type"], inplace=True)
            for course_type in self.course_types:
                time_series[course_type] = 0
            
        for grouping, sub_df in course_dates_in_room.groupby(["start_time", "end_time"]):
            
            # get course time_span
            course_time_mask = (time_series["datetime"] >= grouping[0]) & (time_series["datetime"] <= grouping[1])
            
            # get course_number
            course_number_list = sub_df["course_number"].values
            time_series.loc[course_time_mask, "course_number"] = ",".join(course_number_list)
            course_info = self.dfg.filter_by_courses(self.course_info_table, course_number_list)
              
            # get course features  
            if "registered" in features:
                time_series.loc[course_time_mask, "registered"] = course_info["registered_students"].values.sum()
                
            if "type" in features:
                for course_type in course_info["type"].values:
                    time_series.loc[course_time_mask, course_type] = 1
                
            if "exam" in features:
                time_series.loc[course_time_mask, "exam"] = int(sub_df["exam"].values[0])
                
            if "lecture" in features:
                time_series.loc[course_time_mask, "lecture"] = 1

            if "test" in features:
                time_series.loc[course_time_mask, "test"] = int(sub_df["test"].values[0])
            
            if "tutorium" in features:
                time_series.loc[course_time_mask, "tutorium"] = int(sub_df["tutorium"].values[0])

        return time_series

    ########### Datetime Features ############
    def hod_fourier_series(self, time_series, hourfloat_column):
        time_series["hod1"] = np.sin(2 * np.pi *  (time_series[hourfloat_column]/24))
        time_series["hod2"] = np.cos(2 * np.pi *  (time_series[hourfloat_column]/24))
        return time_series
        
    def dow_fourier_series(self, time_series, day_column):
        time_series["dow1"] = np.sin(2 * np.pi *  (time_series[day_column]/7))
        time_series["dow2"] = np.cos(2 * np.pi *  (time_series[day_column]/7))
        return time_series
    
    def week_fourier_series(self, time_series, week_column):
        time_series["week1"] = np.sin(2 * np.pi *  (time_series[week_column]/52)).astype(np.float64)
        time_series["week2"] = np.cos(2 * np.pi *  (time_series[week_column]/52)).astype(np.float64)
        return time_series
    
    def add_datetime_features(self, time_series, features):
        
        if "hod" in features:
            time_series["hour"] = time_series["datetime"].dt.hour + (time_series["datetime"].dt.minute / 60)
            time_series = self.hod_fourier_series(time_series, "hour")
            time_series.drop(columns=["hour"], inplace=True)
            
        if "dow" in features:
            time_series["dow"] = time_series["datetime"].dt.dayofweek
            time_series = self.dow_fourier_series(time_series, "dow")
            time_series.drop(columns=["dow"], inplace=True)
            
        if "week" in features:
            time_series = self.dfg.derive_week(time_series, "datetime")
            time_series = self.week_fourier_series(time_series, "week")
            time_series.drop(columns=["week"], inplace=True)

        return time_series
    
    
class OccupancyDataset(Dataset):
    
    room_capacities = {0:164, 1:152}
    
    def __init__(self, time_series_dict: dict, hyperparameters:dict, verbose:bool=True):
        """ Constructor for the occupancy dataset
        Task: Convert the cleaned data into a list of samples
        """
        super().__init__()
        
        # the time series must be structured as a dictionary with the room_id as the key
        self.time_series_dict = time_series_dict
        self.room_ids = list(time_series_dict.keys())
        
        # convert frequency to timedelta
        td_freq = pd.to_timedelta(hyperparameters["frequency"])
        
        self.features = set(hyperparameters["features"].split("_"))
        
        if "occcount" in self.features:
            self.occ_feature = "occcount"
        elif "occrate" in self.features:
                self.occ_feature = "occrate"
        else:
            raise ValueError("No target feature found.")
        

        self.exogenous_features = self.features.difference({"occcount", "occrate"})
        if "dow" in self.exogenous_features:
            self.exogenous_features.remove("dow")
            self.exogenous_features = self.exogenous_features.union({"dow1", "dow2"})
        if "hod" in self.exogenous_features:
            self.exogenous_features.remove("hod")
            self.exogenous_features = self.exogenous_features.union({"hod1", "hod2"})
        if "week" in self.exogenous_features:
            self.exogenous_features.remove("week")
            self.exogenous_features = self.exogenous_features.union({"week1", "week2"})
        
        #if self.differencing == "whole":
        #    self.occ_feature = self.occ_feature + "diff"
        #elif self.differencing == "sample":
        #    self.sample_differencing = True
        #    self.occ_feature = self.occ_feature
        #else:
        #    pass
        
        self.differencing = hyperparameters["differencing"]
        self.sample_differencing = False
        
        if self.differencing == "whole":
            
            if self.occ_feature + "1week" in self.features:
                self.exogenous_features = self.exogenous_features.union({self.occ_feature+ "diff" + "1week"})
                self.exogenous_features.remove(self.occ_feature + "1week")
            
            if self.occ_feature + "1day" in self.features:
                self.exogenous_features = self.exogenous_features.union({self.occ_feature+ "diff" + "1day"})
                self.exogenous_features.remove(self.occ_feature + "1day")
                
            self.occ_feature = self.occ_feature + "diff"
            
            
        elif self.differencing == "sample":
            
            self.sample_differencing = True
            
            if self.occ_feature + "1week" in self.features:
                self.exogenous_features = self.exogenous_features.union({self.occ_feature+ "samplediff" + "1week"})
                self.exogenous_features.remove(self.occ_feature + "1week")
            
            if self.occ_feature + "1day" in self.features:
                self.exogenous_features = self.exogenous_features.union({self.occ_feature+ "samplediff" + "1day"})
                self.exogenous_features.remove(self.occ_feature + "1day")
            
        else:
            pass
            
        self.exogenous_features = sorted(list(self.exogenous_features))

        self.include_x_features = hyperparameters["include_x_features"]
        self.x_horizon = hyperparameters["x_horizon"]
        self.y_horizon = hyperparameters["y_horizon"]
        
        self.verbose = verbose
            
        self.samples = []
        for room_id in self.room_ids:
            
            #print("Sample Generation for Room ID: ", room_id)
            occ_time_series = self.time_series_dict[room_id]
            
            # check for holes in the time series -> they break the rolling window
            ts_diff = occ_time_series["datetime"].diff()
            holes = occ_time_series[ts_diff > td_freq]
            
            if holes.empty:
                #print("Check 1: No holes found in the time series.")
                info, X, y_features, y = self.create_samples(occ_time_series, room_id)
                self.samples.extend(list(zip(info, X, y_features, y)))
                #print( "-----------------------------------")
                
            else:
                #print("Check 1: Holes found and taken care of.")
                
                cur_idx = 0
                for hole_idx in holes.index:
                    info, X, y_features, y = self.create_samples(occ_time_series.iloc[cur_idx:hole_idx], room_id)
                    self.samples.extend(list(zip(info, X, y_features, y)))
                    cur_idx = hole_idx
                
                # add the last part of the time series
                info, X, y_features, y = self.create_samples(occ_time_series.iloc[cur_idx:], room_id)
                self.samples.extend(list(zip(info, X, y_features, y)))
                #print( "-----------------------------------")
        
        
        rng = np.random.default_rng(42)
        
        self.corrected_samples = []
        counter_0 = 0
        counter_else = 0
        for info, X, y_features, y in self.samples:
            if (y[:, 0].sum() == 0) and (X[:, 0].sum() == 0):
                if rng.random() < hyperparameters["zero_sample_drop_rate"]:
                    self.corrected_samples.append((info, X, y_features, y))
                    counter_0 += 1

            else:
                self.corrected_samples.append((info, X, y_features, y))
                counter_else += 1
        
        if verbose:
            print("Number of Samples: ", len(self.corrected_samples))        
            print("Number of Samples with y=0: ", counter_0, "Percentage: ", counter_0/len(self.corrected_samples))
            print("Number of Samples with y!=0: ", counter_else, "Percentage: ", counter_else/len(self.corrected_samples))
            print("-----------------")
            
    def create_samples(self, time_series, room_id):
        
        occ_time_series = time_series.copy(deep=True)
        
        X_list = []
        y_list = []
        y_features_list = []
        sample_info = []
    
        # we want to predict the next N steps based on the previous T steps
        window_size = self.x_horizon + self.y_horizon
        for window in occ_time_series.rolling(window=window_size):
            
            if self.sample_differencing:
                window[self.occ_feature+"samplediff"] = window[self.occ_feature].diff(1).combine_first(time_series[self.occ_feature])
                window[self.occ_feature+"samplediff"+"1week"] = window[self.occ_feature + "1week"].diff(1).combine_first(time_series[self.occ_feature + "1week"])
                window[self.occ_feature+"samplediff"+"1day"] = window[self.occ_feature + "1day"].diff(1).combine_first(time_series[self.occ_feature + "1day"])
            
            X_df = window.iloc[:self.x_horizon]
            y_df = window.iloc[self.x_horizon:]

            if self.sample_differencing:
                y = torch.Tensor(y_df[self.occ_feature+"samplediff"].values[:, None])
                X = torch.Tensor(X_df[self.occ_feature+"samplediff"].values[:, None])
                
            else:
                y = torch.Tensor(y_df[self.occ_feature].values[:, None])
                X = torch.Tensor(X_df[self.occ_feature].values[:, None])
            
            if y.numel() != self.y_horizon:
                continue
            else:

                y_features = torch.Tensor(y_df[self.exogenous_features].values)

                if self.include_x_features:
                    X = torch.cat([X, torch.Tensor(X_df[self.exogenous_features].values)], dim=1)
            
            X_list.append(X)
            y_features_list.append(y_features)
            y_list.append(y) 
        
            sample_info.append((room_id, X_df["datetime"], y_df["datetime"], self.exogenous_features, self.room_capacities[room_id]))

        sanity_check_1 = [len(x)==self.x_horizon for x in X_list]
        sanity_check_2 = [len(y)==self.y_horizon for y in y_list]
        
        if (all(sanity_check_1) & all(sanity_check_2)):
            #print("Check 2: All the samples have the correct size.")
            return sample_info, X_list, y_features_list, y_list
        
        else:
            raise ValueError("Sanity Check Failed")
        
    def __len__(self):
        return len(self.corrected_samples)
    
    def __getitem__(self, idx):
        return self.corrected_samples[idx]
    
