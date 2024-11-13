# import DateOffset
from pandas.tseries.offsets import DateOffset
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

national_holidays_2024 = [
    datetime.strptime('01/01/2024', '%m/%d/%Y').date(),  # Neujahr
    datetime.strptime('01/06/2024', '%m/%d/%Y').date(),  # Heilige Drei Könige
    datetime.strptime('04/01/2024', '%m/%d/%Y').date(),  # Ostermontag
    datetime.strptime('05/01/2024', '%m/%d/%Y').date(),  # Staatsfeiertag
    datetime.strptime('05/09/2024', '%m/%d/%Y').date(),  # Christi Himmelfahrt
    datetime.strptime('05/20/2024', '%m/%d/%Y').date(),  # Pfingstmontag
    datetime.strptime('05/30/2024', '%m/%d/%Y').date(),  # Fronleichnam
    datetime.strptime('08/15/2024', '%m/%d/%Y').date(),  # Mariä Himmelfahrt
    datetime.strptime('10/26/2024', '%m/%d/%Y').date(),  # Nationalfeiertag
    datetime.strptime('11/01/2024', '%m/%d/%Y').date(),  # Allerheiligen
    datetime.strptime('12/08/2024', '%m/%d/%Y').date(),  # Mariä Empfängnis
    datetime.strptime('12/25/2024', '%m/%d/%Y').date(),  # Christtag
    datetime.strptime('12/26/2024', '%m/%d/%Y').date(),  # Stefanitag
]

university_holidays_2024 = [
    # 18.05.2024
    datetime.strptime('05/18/2024', '%m/%d/%Y').date(),
    # 21.05.2024
    datetime.strptime('05/21/2024', '%m/%d/%Y').date(),
    # 04.05.2024
    datetime.strptime('05/04/2024', '%m/%d/%Y').date(),
    # 31.05.2024
    datetime.strptime('05/31/2024', '%m/%d/%Y').date(),
]

zwickeltage_2024 = [
    # 10.05.2024
    datetime.strptime('05/10/2024', '%m/%d/%Y').date(),
    # 31.05.2024
    datetime.strptime('05/31/2024', '%m/%d/%Y').date(),
    # 16.08.2024
    datetime.strptime('08/16/2024', '%m/%d/%Y').date(),
]

easter_break_2024 = [
    # easter break: 25.03.2024 - 06.04.2024
    datetime.strptime('03/25/2024', '%m/%d/%Y').date(),
    datetime.strptime('04/06/2024', '%m/%d/%Y').date(),]

summer_break_2024 = [
    # summer break: 1.07.2024 - 30.09.2024
    datetime.strptime('07/01/2024', '%m/%d/%Y').date(),
    datetime.strptime('09/09/2024', '%m/%d/%Y').date(),
]


########### Occupancy Dataset ###########
def load_data(path_to_data_dir, frequency, dfguru, split_by, with_examweek):
    
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    path_to_data = os.path.join(path_to_data_dir, f"freq_{frequency}")
    if with_examweek:
        add_to_string = "_with-examweek"
    else:
        add_to_string = "_without-examweek"
        
    for room_id in [0, 1]:
        for type in ["train", "test", "val"]:
            
            df = dfguru.load_dataframe(
                path_repo=path_to_data, 
                file_name=f"room-{room_id}_{split_by}_{type}_dict" + add_to_string)
            
            if type == "train":
                train_dict[room_id] = df
            elif type == "val":
                val_dict[room_id] = df
            elif type == "test":
                test_dict[room_id] = df
            else:
                raise ValueError("Unknown type.")
            
    return train_dict, val_dict, test_dict
      
def prepare_data(path_to_data_dir, frequency, feature_list, dfguru, rng, helpers_path, split_by, with_examweek):
    
    course_dates_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_dates")
    
    course_info_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_info")

    course_info_data.drop(columns=["room_id"], inplace=True)
    course_info_data.drop_duplicates(inplace=True)

    exam_weeks = [26]
    data_dict = {}
    path_to_occ_data = os.path.join(path_to_data_dir, f"freq_{frequency}")
    for room_id in [0, 1]:
        
        ########## Load Data ##########
        occ_time_series = dfguru.load_dataframe(
            path_repo=path_to_occ_data, 
            file_name=f"room-{room_id}_cleaned_data_29_08", 
        )[:-1]
        
        if not with_examweek:
            occ_time_series = occ_time_series[~occ_time_series["datetime"].dt.isocalendar().week.isin(exam_weeks)]  

        ########## OccFeatureEngineer ##########
        occ_time_series = OccFeatureEngineer(
            occ_time_series, 
            course_dates_data, 
            course_info_data, 
            dfguru,
            frequency,
            helpers_path
        ).derive_features(
            features=feature_list, 
            room_id=room_id
        )
          
        data_dict[room_id] = occ_time_series
        
        
    if with_examweek:
        add_to_string = "_with-examweek"
    else:
        add_to_string = "_without-examweek"
        
    # save unsplit data_dict
    for room_id, df in data_dict.items():
        dfguru.save_to_csv(
            df, 
            f"data/occupancy_forecasting/freq_{frequency}", 
            f"room-{room_id}_unsplit-data-dict" + add_to_string)
    

    train_dict, val_dict, test_dict = train_val_test_split(data_dict, rng, split_by, verbose=True)

    return train_dict, val_dict, test_dict
    
def train_val_test_split(data_dict, rng, split_by, verbose=True):
    
    # randomly exclude chunks of the data
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    total_size = 0
    val_size = 0
    test_size = 0
    train_size = 0
    
    chunk_proportion = 0.025
    test_set_factor = 4
    
    for room_id in data_dict:
         
        if split_by == "random":

            occ_time_series = data_dict[room_id]
            total_size += len(occ_time_series)
            
            # generate chunks with size 0.05 of the data
            index_shift = int(len(occ_time_series) * chunk_proportion)
            indices = np.arange(0, len(occ_time_series), index_shift)
            if (len(occ_time_series)-indices[-1]) < index_shift:
                indices = indices[:-1]
                
            rng.shuffle(indices)
            
            test_slice = int(len(indices) * chunk_proportion * test_set_factor)
            val_indices = indices[:test_slice]
            test_indices = indices[test_slice : 2*test_slice]
            train_indices = indices[2*test_slice:]
            
            ts_val = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in val_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
            ts_test = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in test_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
            ts_train = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in train_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
        
        elif split_by == "random-train-val":
            
            occ_time_series = data_dict[room_id]
            total_size += len(occ_time_series)
            
            # generate chunks with size 0.05 of the data
            index_shift = int(len(occ_time_series) * chunk_proportion)
            indices = np.arange(0, len(occ_time_series), index_shift)
            if (len(occ_time_series)-indices[-1]) < index_shift:
                indices = indices[:-1]
                
            test_size_indices = chunk_proportion * test_set_factor
            test_slice = int(len(indices) * chunk_proportion * test_set_factor)
            
            train_size_indices = 1 - (2*test_size_indices)
            train_slice = int(len(indices) * train_size_indices)
            
            train_val_indices = indices[:train_slice + test_slice]
            test_indices = indices[train_slice + test_slice :]
            
            rng.shuffle(train_val_indices)
            
            train_indices = train_val_indices[:train_slice]
            val_indices = train_val_indices[train_slice:]

            ts_val = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in val_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
            ts_test = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in test_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
            ts_train = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in train_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
           
        elif split_by == "mixed":
            
            occ_time_series = data_dict[room_id]
            total_size += len(occ_time_series)

            # generate chunks with size 0.05 of the data
            index_shift = int(len(occ_time_series) * chunk_proportion)
            indices = np.arange(0, len(occ_time_series), index_shift)
            if (len(occ_time_series)-indices[-1]) < index_shift:
                indices = indices[:-1]
               
               
            test_size_indices = chunk_proportion * test_set_factor
            train_size_indices = 1 - (2*test_size_indices)


            train_slice = int(len(indices) * train_size_indices)
            test_slice = int(len(indices) * test_size_indices)
            
            train_val_indices = indices[ : train_slice +test_slice//2].copy()
            rng.shuffle(train_val_indices)
            
            train_indices = train_val_indices[ : train_slice]
            val_indices_2 = train_val_indices[train_slice : ]
            
            val_indices = indices[train_slice+test_slice//2 : train_slice+test_slice]
            val_indices = np.concatenate([val_indices, val_indices_2])
            
            test_indices = indices[train_slice+test_slice : ]

            ts_val = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in val_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
            ts_test = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in test_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
            ts_train = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in train_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
           
        elif split_by == "time":
            
            occ_time_series = data_dict[room_id]
            total_size += len(occ_time_series)

            # generate chunks with size 0.05 of the data
            index_shift = int(len(occ_time_series) * chunk_proportion)
            indices = np.arange(0, len(occ_time_series), index_shift)
            if (len(occ_time_series)-indices[-1]) < index_shift:
                indices = indices[:-1]
               
            test_size_indices = chunk_proportion * test_set_factor
            train_size_indices = 1 - (2*test_size_indices)

            train_slice = int(len(indices) * train_size_indices)
            test_slice = int(len(indices) * test_size_indices)
            
            train_indices = indices[ : train_slice]
            val_indices = indices[train_slice : train_slice+test_slice]
            
            test_indices = indices[train_slice+test_slice : ]

            ts_val = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in val_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
            ts_test = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in test_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)
            ts_train = occ_time_series.iloc[np.array([np.arange(x, x+index_shift) for x in train_indices]).flatten()].sort_values(by="datetime").reset_index(drop=True)

        else:
            
            raise ValueError("Unknown split method.")
        
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
    
    course_features = {"maxocccount", "maxoccrate" ,"maxoccrateestimate", "maxocccountestimate",
                       "coursenumber", "exam",  "test", "tutorium", "cancelled", 
                       "lecture", "lecturerampbefore", "lecturerampafter",
                       "registered", "type", "studyarea", "ects", "level"}
    datetime_features = {"dow", "hod", "week", "holiday", "zwickltag"}
    general_features = {"occcount", "occrate"}
    weather_features = {"weather"}
    shift_features = {"occcount1week", "occrate1week", "occcount1day", "occrate1day"}
    permissible_features = course_features.union(datetime_features)
    permissible_features = permissible_features.union(general_features)
    permissible_features = permissible_features.union(weather_features)
    permissible_features = permissible_features.union(shift_features)
      
    def __init__(self, cleaned_occ_data, course_dates_data, course_info_data, 
                 dfguru, frequency, helpers_path):

        self.occ_time_series = cleaned_occ_data
        min_timestamp = self.occ_time_series["datetime"].min().replace(hour=0, minute=0, second=0, microsecond=0)
        max_timestamp = self.occ_time_series["datetime"].max().replace(hour=0, minute=0, second=0, microsecond=0) + DateOffset(days=1)
        
        ##### Define Constants and DFGuru #####
        self.helpers_path = os.path.join(helpers_path, "helpers_occpred.json")
        self.dfg = dfguru
        self.frequency = frequency
        
        ####### Load Data ########
        self.course_dates_table = dfguru.filter_by_timestamp(
            dataframe = course_dates_data,
            start_time = min_timestamp,
            end_time = max_timestamp,
            time_column = "start_time"
        )
        self.course_numbers = self.course_dates_table["course_number"].unique()
        
        self.course_info_table = dfguru.filter_by_courses(
            dataframe = course_info_data,
            course_numbers = self.course_numbers
        )

        ####### Define Helpers ########
        # course types
        self.course_types = ["VL", "UE", "KS"]
        self.type_mapping = {
            "VL": "VL",
            "VO": "VL",
            "KO": "VL",
            "UE": "UE",
            "AG": "UE",
            "IK": "UE",
            "PR": "UE",
            "PS": "UE",
            "KS": "KS",
            "VU": "KS",
            "KV": "KS",
            "RE": "KS",
            "UV": "KS"
        }       
        
        self.course_info_table["level"].fillna("None_level", inplace=True)
        self.course_info_table["study_area"].fillna("None_sa", inplace=True)

        # study areas and levels
        self.study_areas = self.course_info_table["study_area"].unique()
        self.levels = self.course_info_table["level"].unique()
                      
        ###### Load & Handle Weather Data ######
        # weather data
        self.weather = dfguru.load_dataframe("data/weather", "measurements_10min", dtypes=False)
        
        # most important columns: tl_max, tl_min, p_mittel, vv_mittel, ff_max, rfb_mittel, rr, so_h
        self.weather_columns = ["time", "tl", "p", "ff", "ffx", "rf", "rr", "so"]
        self.weather = self.weather[self.weather_columns]

        self.weather["time"] = pd.to_datetime(self.weather["time"], format="%Y-%m-%dT%H:%M+00:00")
        
        # set time as index
        self.weather.set_index("time", inplace=True)
        
        self.weather["ff"] = self.weather["ff"].fillna(method="ffill")
        self.weather["ffx"] = self.weather["ffx"].fillna(method="ffill")
    
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
        
        # add weather features
        weather_features = self.weather_features.intersection(feature_set)
        if weather_features:
            occ_time_series = self.add_weather_features(occ_time_series, weather_features)

        
        ###### save all helper data ######
        if os.path.exists(self.helpers_path):
            with open(self.helpers_path, "r") as f:
                helper_dict = json.load(f)   
            
        else:
            helper_dict = {
                "course_types": set(),
                "course_numbers": set(),
                "study_areas": set(),
                "levels": set(),
                "weather_columns": set(),
                "columns_to_normalize":{
                    "occcount":{"min":np.inf, "max":-np.inf},
                    "occcountdiff":{"min":np.inf, "max":-np.inf},
                    "maxocccount":{"min":np.inf, "max":-np.inf},
                    "registered":{"min":np.inf, "max":-np.inf},
                    "ects":{"min":np.inf, "max":-np.inf},
                    "tl":{"min":np.inf, "max":-np.inf},
                    "p":{"min":np.inf, "max":-np.inf},
                    "ff":{"min":np.inf, "max":-np.inf},
                    "ffx":{"min":np.inf, "max":-np.inf},
                    "rf":{"min":np.inf, "max":-np.inf},
                    "rr":{"min":np.inf, "max":-np.inf},
                    "so":{"min":np.inf, "max":-np.inf}
                }
            }
        
        
        helper_dict["course_types"] = list(set(helper_dict["course_types"]).union(set(self.course_types)))
        helper_dict["course_numbers"] = list(set(helper_dict["course_numbers"]).union(set(occ_time_series["coursenumber"].unique())))
        helper_dict["study_areas"] = list(set(helper_dict["study_areas"]).union(set(self.study_areas)))
        helper_dict["levels"] = list(set(helper_dict["levels"]).union(set(self.levels)))
        helper_dict["weather_columns"] = list(set(helper_dict["weather_columns"]).union(set(self.weather_columns[1:])))

        columns_to_normalize = [
             'occcount', 'occcountdiff',
             'maxocccount', 'registered', 'ects', 
             'tl', 'p', 'ff', 'ffx', 'rf', 'rr', 'so'
        ]
        for col in columns_to_normalize:
            
            col_entry = helper_dict["columns_to_normalize"][col]
            min, max = float(occ_time_series[col].min()), float(occ_time_series[col].max())
            
            if col_entry["min"] > min:
                col_entry["min"] = min

            if col_entry["max"] < max:
                col_entry["max"] = max
                
        # save auxillary data
        with open(file=self.helpers_path, mode="w") as file:
            json.dump(helper_dict, file, indent=4)

        return occ_time_series
    
    ########### General Features ############
    def add_shift_features(self, time_series, features):
        # initialize all features to 0
        for feature in features:
            time_series[feature] = 0
            
        if "occcount1week" in features:
            
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
        
        # get some constants
        ramp_duration = pd.to_timedelta("15min")
        room_capa = course_dates_in_room["room_capacity"].unique()
        
        # load lecture occrate extimates
        max_occrate_estimates = np.load("data/lecture_maxoccrate_estimates.npy", allow_pickle=True)
        pd_maxoccrate = pd.DataFrame(max_occrate_estimates, columns=["course_number", "starttime", "roomid", "maxoccrateestimate", "maxoccrate"])
        

        # initialize all features to 0
        for feature in features:
            time_series[feature] = 0
            
        if "type" in features:
            time_series.drop(columns=["type"], inplace=True)
            for course_type in self.course_types:
                time_series[course_type] = 0
        
        if "studyarea" in features:
            time_series.drop(columns=["studyarea"], inplace=True)
            for study_area in self.study_areas:
                time_series[study_area] = 0
        
        
        if "level" in features:
            time_series.drop(columns=["level"], inplace=True)
            for level in self.levels:
                time_series[level] = 0

        time_series["coursenumber"] = ''
        

        
        for grouping, sub_df in course_dates_in_room.groupby(["start_time", "end_time"]):
            
            # get course time_span
            course_time_mask = (time_series["datetime"] >= grouping[0]) & (time_series["datetime"] <= grouping[1])
            
            # get course_number
            course_number_list = sub_df["course_number"].values
            time_series.loc[course_time_mask, "coursenumber"] = ",".join(sorted(course_number_list))
            course_info = self.dfg.filter_by_courses(self.course_info_table, course_number_list)

            if "maxocccount" in features:
                time_series.loc[course_time_mask, "maxocccount"] = time_series.loc[course_time_mask, "occcount"].max()
            
            if "maxoccrate" in features:
                time_series.loc[course_time_mask, "maxoccrate"] = time_series.loc[course_time_mask, "occrate"].max()
                
            if "maxocccountestimate" in features:
                
                masked_df = pd_maxoccrate[(pd_maxoccrate["starttime"] == grouping[0]) & (pd_maxoccrate["roomid"] == room_id)]
                
                if masked_df.empty:
                    time_series.loc[course_time_mask, "maxoccrateestimate"] = -1
                else:
                    registered_students = course_info["registered_students"].values           
                    max_occount_estimate = int((masked_df["maxoccrateestimate"].values * registered_students).sum())
                    time_series.loc[course_time_mask, "maxoccrateestimate"] = max_occount_estimate
                
            if "maxoccrateestimate" in features:
                
                masked_df = pd_maxoccrate[(pd_maxoccrate["starttime"] == grouping[0]) & (pd_maxoccrate["roomid"] == room_id)]
                
                if masked_df.empty:
                    time_series.loc[course_time_mask, "maxoccrateestimate"] = -1
                else:
                    registered_students = course_info["registered_students"].values
                    max_occount_estimate = (masked_df["maxoccrateestimate"].values * registered_students).sum()
                    max_occrate_estimate = max_occount_estimate / room_capa
                    time_series.loc[course_time_mask, "maxoccrateestimate"] = float(max_occrate_estimate)
                
            # get course features  
            if "registered" in features:
                time_series.loc[course_time_mask, "registered"] = course_info["registered_students"].values.sum()
                
            if "type" in features:
                for course_type in course_info["type"].values:
                    time_series.loc[course_time_mask, self.type_mapping[course_type]] = 1

            if "studyarea" in features:
                for study_area in course_info["study_area"].values:
                    time_series.loc[course_time_mask, study_area] = 1
            
            if "level" in features:
                for level in course_info["level"].values:
                    time_series.loc[course_time_mask, level] = 1
            
            if "ects" in features:
                time_series.loc[course_time_mask, "ects"] = float(".".join(course_info["ects"].values[0].split(",")))

            if "exam" in features:
                time_series.loc[course_time_mask, "exam"] = int(sub_df["exam"].values[0])
                
            if "lecture" in features:
                time_series.loc[course_time_mask, "lecture"] = 1
                
            if "test" in features:
                time_series.loc[course_time_mask, "test"] = int(sub_df["test"].values[0])
            
            if "tutorium" in features:
                time_series.loc[course_time_mask, "tutorium"] = int(sub_df["tutorium"].values[0])
                
            if "cancelled" in features:
                time_series.loc[course_time_mask, "cancelled"] = int(sub_df["cancelled"].values[0])
            
            if "lecturerampbefore" in features:
                start_minus_15 = grouping[0] - ramp_duration
                ramp_up_mask = (time_series["datetime"] >= start_minus_15) & (time_series["datetime"] < grouping[0])
                ramp_up_fraction = (time_series["datetime"][ramp_up_mask] - start_minus_15) / ramp_duration
                time_series.loc[ramp_up_mask, "lecturerampbefore"] = ramp_up_fraction
                
                
                #if "maxoccrate" in features:
                #    time_series.loc[ramp_up_mask, "maxoccrate"] = time_series.loc[course_time_mask, "occrate"].max()
                    
                #if "maxoccrateestimate" in features:
                #    time_series.loc[ramp_up_mask, "maxoccrateestimate"] = time_series.loc[course_time_mask, "maxoccrateestimate"].max()
                    
                #if "maxocccount" in features:
                #    time_series.loc[ramp_up_mask, "maxocccount"] = time_series.loc[course_time_mask, "occcount"].max()
                
                #if "maxocccountestimate" in features:
                #    time_series.loc[ramp_up_mask, "maxocccountestimate"] = time_series.loc[course_time_mask, "maxocccountestimate"].max()
                         
            if "lecturerampafter" in features:
                end_plus_15 = grouping[1] + ramp_duration
                ramp_down_mask = (time_series["datetime"] > grouping[1]) & (time_series["datetime"] <= end_plus_15)
                ramp_down_fraction = (end_plus_15 - time_series["datetime"][ramp_down_mask]) / ramp_duration
                time_series.loc[ramp_down_mask, "lecturerampafter"] = ramp_down_fraction

        return time_series

    ########### Weather Features ############
    def add_weather_features(self, time_series, features):

        # resample the weather data to the frequency of the occupancy data       
        self.weather = self.weather.resample(self.frequency).ffill()
        self.weather = self.weather[(self.weather.index >= time_series["datetime"].min()) & (self.weather.index <= time_series["datetime"].max())]
        
        weathercols = self.weather_columns[1:]
        # init weather features
        for weathercol in weathercols:
            time_series[weathercol] = 0

        # add weather features
        for weathercol in weathercols:
            time_series[weathercol] = self.weather[weathercol].values

        return time_series

    ########### Datetime Features ############
    def hod_fourier_series(self, time_series, hourfloat_column):
        time_series["hod1"] = 0.5 * np.sin(2 * np.pi *  (time_series[hourfloat_column]/24)) + 0.5
        time_series["hod2"] = 0.5 * np.cos(2 * np.pi *  (time_series[hourfloat_column]/24)) + 0.5
        return time_series
        
    def dow_fourier_series(self, time_series, day_column):
        time_series["dow1"] = 0.5 * np.sin(2 * np.pi *  (time_series[day_column]/7)) + 0.5
        time_series["dow2"] = 0.5 * np.cos(2 * np.pi *  (time_series[day_column]/7)) + 0.5
        return time_series
    
    def week_fourier_series(self, time_series, week_column):
        time_series["week1"] = 0.5 * np.sin(2 * np.pi *  (time_series[week_column]/52)).astype(np.float64) + 0.5
        time_series["week2"] = 0.5 * np.cos(2 * np.pi *  (time_series[week_column]/52)).astype(np.float64) + 0.5
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
            
        if "holiday" in features:
            time_series["holiday"] = 0
            time_series["day"] = time_series["datetime"].dt.date
            time_series = self.derive_holiday(time_series, national_holidays_2024 + university_holidays_2024, "day", "holiday")
            time_series["holiday"] = time_series["holiday"].astype(int)
            time_series.drop(columns=["day"], inplace=True)
            
        if "zwickltag" in features:
            time_series["zwickltag"] = 0
            time_series["day"] = time_series["datetime"].dt.date
            time_series = self.derive_holiday(time_series, zwickeltage_2024, "day", "zwickltag")
            time_series["zwickltag"] = time_series["zwickltag"].astype(int)
            time_series.drop(columns=["day"], inplace=True)

        return time_series    
    
    def derive_holiday(self, dataframe, holiday_dates, date_column, out_column):
        dataframe[out_column] = dataframe[date_column].isin(holiday_dates)
        return dataframe 
    
    #def weekday_feature(self, time_series):
    #    print(time_series)
    #    #time_series[""] = time_series["datetime"].dt.date
        
    #    raise NotImplementedError("Not implemented yet.")
    #    return time_series
           
class OccupancyDataset(Dataset):
    
    room_capacities = {0:164, 1:152}
    
    def __init__(self, time_series_dict: dict, hyperparameters:dict, path_to_helpers:str, validation:bool, verbose:bool=True):
        """ Constructor for the occupancy dataset
        Task: Convert the cleaned data into a list of samples
        """
        super().__init__()
        
        self.rng = np.random.default_rng(42)
        self.dataset_mode = hyperparameters["dataset_mode"]
        
        # the time series must be structured as a dictionary with the room_id as the key
        self.time_series_dict = time_series_dict
        self.room_ids = list(time_series_dict.keys())
        
        self.hyperparameters = hyperparameters

        ############ Handle Features ############
        self.features = set(hyperparameters["features"].split("_"))
        
        if "maxocccountestimate" in self.features:
            raise ValueError("Feature 'maxocccountestimate' is not allowed.")
        
        # derive main feature
        self.occ_feature = self.handle_occ_feature(self.features)

        # derive exogenous features
        self.exogenous_features = self.features.difference({"occcount", "occrate"})
        # derive exogenous time features
        self.exogenous_features = self.handle_time_features(self.exogenous_features)
        # derive differencing
        self.differencing = hyperparameters["differencing"]
        self.occ_feature, self.exogenous_features, self.sample_differencing = self.handle_differencing_features(self.differencing, self.features, self.occ_feature, self.exogenous_features)

            
        helper_file = os.path.join(path_to_helpers, "helpers_occpred.json")
        with open(helper_file, "r") as f:
            self.helper = json.load(f)       


        if "type" in self.exogenous_features:
            self.exogenous_features.remove("type")
            self.exogenous_features = self.exogenous_features.union(self.helper["course_types"])
        
        self.extract_coursenumber = False
        if "coursenumber" in self.exogenous_features:
            self.exogenous_features.remove("coursenumber")
            self.extract_coursenumber = True
            self.course_numbers = self.helper["course_numbers"]
            self.coursenr_lookup = dict([(x,i) for i,x in enumerate(self.course_numbers)])
            
        if "studyarea" in self.exogenous_features:
            self.exogenous_features.remove("studyarea")
            self.exogenous_features = self.exogenous_features.union(self.helper["study_areas"])
        
        if "level" in self.exogenous_features:
            self.exogenous_features.remove("level")
            self.exogenous_features = self.exogenous_features.union(self.helper["levels"])
        
        if "weather" in self.exogenous_features:
            self.exogenous_features.remove("weather")
            self.exogenous_features = self.exogenous_features.union(self.helper["weather_columns"])

        # sort features
        self.exogenous_features = sorted(list(self.exogenous_features))
    
        ############ Derive some helper variables ############
        self.include_x_features = hyperparameters["include_x_features"]
        self.x_horizon = hyperparameters["x_horizon"]
        self.y_horizon = hyperparameters["y_horizon"]
        self.verbose = verbose       
         
        ############ Process Data ############
        # convert frequency to timedelta
        self.td_freq = pd.to_timedelta(hyperparameters["frequency"])
        # process data
        self.samples = self.process_data_dictionaries(self.dataset_mode)
        # correct samples
        if not validation:
            if self.dataset_mode == "normal":
                self.correct_samples(self.samples, verbose=self.verbose)

    ############ Feature Functions ############ 
    def handle_occ_feature(self, features):
            if "occcount" in features:
                return "occcount"
            elif "occrate" in features:
                return "occrate"
            elif "occpresence" in features:
                return "occpresence"
            else:
                raise ValueError("No target feature found.")

    def handle_time_features(self, exo_features):
        
        copied_exo_features = exo_features.copy()
        
        if "dow" in copied_exo_features:
            copied_exo_features.remove("dow")
            copied_exo_features = copied_exo_features.union({"dow1", "dow2"})
            
        if "hod" in copied_exo_features:
            copied_exo_features.remove("hod")
            copied_exo_features = copied_exo_features.union({"hod1", "hod2"})
            
        if "week" in copied_exo_features:
            copied_exo_features.remove("week")
            copied_exo_features = copied_exo_features.union({"week1", "week2"})
            
        return copied_exo_features
    
    def handle_differencing_features(self, differencing, features, occ_feature, exo_features):
            
        copied_exo_features = exo_features.copy()
        
        sample_differencing = False
        
        if differencing == "whole":
        
            if occ_feature + "1week" in features:
                copied_exo_features = copied_exo_features.union({occ_feature + "diff" + "1week"})
                copied_exo_features.remove(occ_feature + "1week")
            
            if occ_feature + "1day" in features:
                copied_exo_features = copied_exo_features.union({occ_feature + "diff" + "1day"})
                copied_exo_features.remove(occ_feature + "1day")
                
            occ_feature = occ_feature + "diff"
            
        elif differencing == "sample":
            
            sample_differencing = True
            
            if occ_feature + "1week" in features:
                copied_exo_features = copied_exo_features.union({occ_feature + "samplediff" + "1week"})
                copied_exo_features.remove(occ_feature + "1week")
            
            if occ_feature + "1day" in features:
                copied_exo_features = copied_exo_features.union({occ_feature + "samplediff" + "1day"})
                copied_exo_features.remove(occ_feature + "1day")
            
        else:
            pass
        
        return occ_feature, copied_exo_features, sample_differencing

    ############ Process Data Functions ############
    def process_data_dictionaries(self, mode):
        
        if mode=="unlimited":
            sampling_function = self.create_samples_unlimited
        elif mode=="dayahead":
            sampling_function = self.create_samples_dayahead
        elif mode=="normal":
            sampling_function = self.create_samples_normal
        else:
            raise ValueError("Unknown mode.")
        
        samples = []
        for room_id in self.room_ids:

            occ_time_series = self.time_series_dict[room_id]
            
            # normalize samples
            for col, values in self.helper["columns_to_normalize"].items():
                
                if col in self.exogenous_features:
                    occ_time_series[col] = (occ_time_series[col] - values["min"]) / (values["max"] - values["min"])
      
                elif col in self.occ_feature:
                    occ_time_series[col] = (occ_time_series[col] - values["min"]) / (values["max"] - values["min"])
     
                else:
                    pass

            ts_diff = occ_time_series["datetime"].diff()
            holes = occ_time_series[ts_diff > self.td_freq]
            
            if holes.empty:

                info, X, y_features, y = sampling_function(occ_time_series, room_id)
                samples.extend(list(zip(info, X, y_features, y)))

            else:
                
                cur_idx = 0
                for hole_idx in holes.index:
                    info, X, y_features, y = sampling_function(occ_time_series.iloc[cur_idx:hole_idx], room_id)
                    samples.extend(list(zip(info, X, y_features, y)))
                    cur_idx = hole_idx
                
                # add the last part of the time series
                info, X, y_features, y = sampling_function(occ_time_series.iloc[cur_idx:], room_id)
                samples.extend(list(zip(info, X, y_features, y)))
                
        return samples
          
    def create_samples_normal(self, time_series, room_id):
        
        occ_time_series = time_series.copy(deep=True)
        
        X_list = []
        y_list = []
        y_features_list = []
        sample_info = []
        
        if self.occ_feature == "occpresence":
            # occpresence is a binary feature -> 1 if occrate != 0, 0 otherwise
            occ_time_series["occpresence"] = occ_time_series["occrate"].apply(lambda x: 1 if x > 0 else 0)
            occ_time_series["occpresence1week"] = occ_time_series["occrate1week"].apply(lambda x: 1 if x > 0 else 0)
            occ_time_series["occpresence1day"] = occ_time_series["occrate1day"].apply(lambda x: 1 if x > 0 else 0)

        if ("maxoccrate" in self.exogenous_features) & (self.hyperparameters["differencing"] == "whole"):
            occ_time_series["maxoccrate"] = occ_time_series["maxoccrate"].diff(1).combine_first(occ_time_series["maxoccrate"])
            
        # we want to predict the next N steps based on the previous T steps
        window_size = self.x_horizon + self.y_horizon
        for window in occ_time_series.rolling(window=window_size):
            
            if self.sample_differencing:
                window[self.occ_feature+"samplediff"] = window[self.occ_feature].diff(1).combine_first(window[self.occ_feature])
                window[self.occ_feature+"samplediff"+"1week"] = window[self.occ_feature + "1week"].diff(1).combine_first(window[self.occ_feature + "1week"])
                window[self.occ_feature+"samplediff"+"1day"] = window[self.occ_feature + "1day"].diff(1).combine_first(window[self.occ_feature + "1day"])
            
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
                    
                if self.extract_coursenumber:
                    
                    X_course = X_df["coursenumber"].fillna("")
                    X_course = X_course.apply(lambda x: "{:.3f}".format(x) if type(x) == float else x)
                    X_course = X_course.apply(lambda x: self.coursenr_lookup[x])
                    X_course = torch.Tensor(X_course.values)
                    
                    y_course = y_df["coursenumber"].fillna("")
                    y_course = y_course.apply(lambda x: "{:.3f}".format(x) if type(x) == float else x)
                    y_course = y_course.apply(lambda x: self.coursenr_lookup[x])
                    y_course = torch.Tensor(y_course.values)    
                    
                else:
                    X_course = None
                    y_course = None    

            
            X_list.append(X)
            y_features_list.append(y_features)
            y_list.append(y) 
        
            sample_info.append((room_id, X_df["datetime"], y_df["datetime"], self.exogenous_features, self.room_capacities[room_id], (X_course, y_course)))

        sanity_check_1 = [len(x)==self.x_horizon for x in X_list]
        sanity_check_2 = [len(y)==self.y_horizon for y in y_list]
        
        if (all(sanity_check_1) & all(sanity_check_2)):
            #print("Check 2: All the samples have the correct size.")
            return sample_info, X_list, y_features_list, y_list
        
        else:
            raise ValueError("Sanity Check Failed")
        
    def create_samples_unlimited(self, time_series, room_id):
        
        time_series = time_series.copy(deep=True)
        
        X_list = []
        y_list = []
        y_features_list = []
        sample_info = []
        
        # we want to predict the next N steps based on the previous T steps
        if self.occ_feature == "occpresence":
            # occpresence is a binary feature -> 1 if occrate != 0, 0 otherwise
            time_series["occpresence"] = time_series["occrate"].apply(lambda x: 1 if x > 0 else 0)
            time_series["occpresence1week"] = time_series["occrate1week"].apply(lambda x: 1 if x > 0 else 0)
            time_series["occpresence1day"] = time_series["occrate1day"].apply(lambda x: 1 if x > 0 else 0)
             
        if self.sample_differencing:
            time_series[self.occ_feature+"samplediff"] = time_series[self.occ_feature].diff(1).combine_first(time_series[self.occ_feature])
            time_series[self.occ_feature+"samplediff"+"1week"] = time_series[self.occ_feature + "1week"].diff(1).combine_first(time_series[self.occ_feature + "1week"])
            time_series[self.occ_feature+"samplediff"+"1day"] = time_series[self.occ_feature + "1day"].diff(1).combine_first(time_series[self.occ_feature + "1day"])
            
        X_df = time_series.iloc[:self.x_horizon]
        y_df = time_series.iloc[self.x_horizon:]
        
        if self.sample_differencing:
            y = torch.Tensor(y_df[self.occ_feature+"samplediff"].values[:, None])
            X = torch.Tensor(X_df[self.occ_feature+"samplediff"].values[:, None])
            
        else:
            y = torch.Tensor(y_df[self.occ_feature].values[:, None])
            X = torch.Tensor(X_df[self.occ_feature].values[:, None])
            

        y_features = torch.Tensor(y_df[self.exogenous_features].values)

        if self.include_x_features:
            X = torch.cat([X, torch.Tensor(X_df[self.exogenous_features].values)], dim=1)

        if self.extract_coursenumber:
            
            X_course = X_df["coursenumber"].fillna("")
            X_course = X_course.apply(lambda x: "{:.3f}".format(x) if type(x) == float else x)
            X_course = X_course.apply(lambda x: self.coursenr_lookup[x])
            X_course = torch.Tensor(X_course.values)
            
            y_course = y_df["coursenumber"].fillna("")
            y_course = y_course.apply(lambda x: "{:.3f}".format(x) if type(x) == float else x)
            y_course = y_course.apply(lambda x: self.coursenr_lookup[x])
            y_course = torch.Tensor(y_course.values)    
            
        else:
            X_course = None
            y_course = None    

        X_list.append(X)
        y_features_list.append(y_features)
        y_list.append(y) 
        sample_info.append((room_id, X_df["datetime"], y_df["datetime"], self.exogenous_features, self.room_capacities[room_id], (X_course, y_course)))

        return sample_info, X_list, y_features_list, y_list

    def create_samples_dayahead(self, time_series, room_id):
        
        occ_time_series = time_series.copy(deep=True)
        
        X_list = []
        y_list = []
        y_features_list = []
        sample_info = []

        if self.occ_feature == "occpresence":
            # occpresence is a binary feature -> 1 if occrate != 0, 0 otherwise
            occ_time_series["occpresence"] = occ_time_series["occrate"].apply(lambda x: 1 if x > 0 else 0)
            occ_time_series["occpresence1week"] = occ_time_series["occrate1week"].apply(lambda x: 1 if x > 0 else 0)
            occ_time_series["occpresence1day"] = occ_time_series["occrate1day"].apply(lambda x: 1 if x > 0 else 0)


        if ("maxoccrate" in self.exogenous_features) & (self.hyperparameters["differencing"] == "whole"):
            occ_time_series["maxoccrate"] = occ_time_series["maxoccrate"].diff(1).combine_first(occ_time_series["maxoccrate"])
            

        # we want to predict the nex day based on the previous T steps
        occ_time_series["day"] = occ_time_series["datetime"].dt.date
        
        i = 0
        
        for grouping, sub_df in occ_time_series.groupby("day"):
            if i == 0:
                i += 1
                continue

            X_df = occ_time_series[occ_time_series["day"] == (grouping - pd.Timedelta("1d"))]

            X_df = X_df.iloc[-self.x_horizon:]

            if len(X_df) < self.x_horizon:
                continue
            
            y_df = sub_df
            
            # we want to predict the next N steps based on the previous T steps
            if self.sample_differencing:
                raise NotImplementedError("Differencing not implemented for dayahead mode.")
                time_series[self.occ_feature+"samplediff"] = time_series[self.occ_feature].diff(1).combine_first(time_series[self.occ_feature])
                time_series[self.occ_feature+"samplediff"+"1week"] = time_series[self.occ_feature + "1week"].diff(1).combine_first(time_series[self.occ_feature + "1week"])
                time_series[self.occ_feature+"samplediff"+"1day"] = time_series[self.occ_feature + "1day"].diff(1).combine_first(time_series[self.occ_feature + "1day"])
                y = torch.Tensor(y_df[self.occ_feature+"samplediff"].values[:, None])
                X = torch.Tensor(X_df[self.occ_feature+"samplediff"].values[:, None])
                
            y = torch.Tensor(y_df[self.occ_feature].values[:, None])
            X = torch.Tensor(X_df[self.occ_feature].values[:, None])         
            y_features = torch.Tensor(y_df[self.exogenous_features].values)
            
            if self.include_x_features:
                X = torch.cat([X, torch.Tensor(X_df[self.exogenous_features].values)], dim=1)
            
            if self.extract_coursenumber:
                
                X_course = X_df["coursenumber"].fillna("")
                X_course = X_course.apply(lambda x: "{:.3f}".format(x) if type(x) == float else x)
                X_course = X_course.apply(lambda x: self.coursenr_lookup[x])
                X_course = torch.Tensor(X_course.values)
                
                y_course = y_df["coursenumber"].fillna("")
                y_course = y_course.apply(lambda x: "{:.3f}".format(x) if type(x) == float else x)
                y_course = y_course.apply(lambda x: self.coursenr_lookup[x])
                y_course = torch.Tensor(y_course.values)    
                
            else:
                X_course = None
                y_course = None    

            X_list.append(X)
            y_features_list.append(y_features)
            y_list.append(y) 
            sample_info.append((room_id, X_df["datetime"], y_df["datetime"], self.exogenous_features, self.room_capacities[room_id], (X_course, y_course)))
                                         
        return sample_info, X_list, y_features_list, y_list  
    
    def correct_samples(self, samples, verbose=True):
        
        
        # derive weights for samples 
        
        one_hour = int(pd.Timedelta("1h")/self.td_freq)
        
        corrected_samples = []
        counter_0 = 0
        counter_else = 0
        
        #exam_idx = self.exogenous_features.index("exam")
        #test_idx = self.exogenous_features.index("test")
        #tutorium_idx = self.exogenous_features.index("tutorium")
        
        #exam_dict = dict([(i,0) for i in range(self.y_horizon+1)])
        #tutorium_dict = dict([(i,0) for i in range(self.y_horizon+1)])
        #test_dict = dict([(i,0) for i in range(self.y_horizon+1)])
        
        class_dict = dict([(i,"") for i in range(len(samples))])
        
        class_counts = {"zero":0, "else":0, "negative":0, "positive":0}
        for i,(info, X, y_features, y) in enumerate(samples):
            
            #exam = y_features[:, exam_idx]
            #test = y_features[:, test_idx]
            #tuturium = y_features[:, tutorium_idx]
            if all((y[:, 0] == 0)) and all((X[-1*one_hour:, 0] == 0)):
                class_dict[i] = "zero"
                
            else:
                if self.differencing == "whole":
                    if all(y[:, 0] <= 0):
                        class_dict[i] = "negative"
                    elif all(y[:, 0] >= 0):
                        class_dict[i] = "positive"
                    else:
                        class_dict[i] = "else"
                        
                else:
                    class_dict[i] = "else"
                
            class_counts[class_dict[i]] += 1
            
            #exam_dict[int(exam.sum())] += 1
            #tutorium_dict[int(tuturium.sum())] += 1
            #test_dict[int(test.sum())] += 1
            
            # old version
            #if (y[:, 0].sum() == 0) and (X[-one_hour:, 0].sum() == 0):
            #    if self.rng.random() < self.hyperparameters["zero_sample_drop_rate"]:
            #        corrected_samples.append((info, X, y_features, y))
            #        counter_0 += 1

            #else:
            #    corrected_samples.append((info, X, y_features, y))
            #    counter_else += 1
            
        class_counts["zero"] = class_counts["zero"]*0.05
        total = sum(class_counts.values())
        
        class_weights = dict([(i, v/total) for i,v in class_counts.items()])

        sample_weights = np.zeros(len(samples))
        for i, class_label in class_dict.items():
            sample_weights[i] = class_weights[class_label]
        
        self.sample_weights = sample_weights / sample_weights.sum()
        
        #if verbose:
        #    print("Number of Samples: ", len(corrected_samples))        
        #    print("Number of Samples with y=0: ", counter_0, "Percentage: ", counter_0/len(corrected_samples))
        #    print("Number of Samples with y!=0: ", counter_else, "Percentage: ", counter_else/len(corrected_samples))
        #    print("-----------------")

       
    ############ Dataset Functions ############
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
