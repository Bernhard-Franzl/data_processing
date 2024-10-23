# import DateOffset
from pandas.tseries.offsets import DateOffset
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
         
########### Lecture Dataset ###########
def load_data(path_to_data_dir, dfguru, split_by):

    traindf = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name=f"lecture_train_{split_by}")
    
    valdf = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name=f"lecture_val_{split_by}")
    
    testdf = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name=f"lecture_test_{split_by}")
    
    return traindf, valdf, testdf
    
def prepare_data(path_to_data_dir, feature_list, dfguru, rng, split_by, test):
    
    course_dates_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_dates")
    
    course_info_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_info")
    
    data_dict = {}
    path_to_occ_data = os.path.join(path_to_data_dir, f"freq_1min")
    for room_id in [0, 1]:
        
        ########## Load Data ##########
        occ_time_series = dfguru.load_dataframe(
            path_repo=path_to_occ_data, 
            file_name=f"room-{room_id}_cleaned_data_29_08", 
        )[:-1]
        data_dict[room_id] = occ_time_series
        
    
    ########## OccFeatureEngineer ##########

    
    lfe = LectureFeatureEngineer(
        data_dict, 
        course_dates_data, 
        course_info_data, 
        dfguru,
        helpers_path=f"data/helpers_lecture_{split_by}.json"
    )
    course_dates_df = lfe.derive_features(
        features=feature_list, 
    )
        
    datasets, indices = train_val_test_split(course_dates_df, rng, split_by, test, verbose=True)
    
    return datasets, indices

def train_val_test_split(course_dates, rng, split_by, test, verbose=True):
    
    # randomly exclude chunks of the data
    #train_dict = {}
    #val_dict = {}
    #test_dict = {}
    
    #total_size = 0
    #val_size = 0
    #test_size = 0
    #train_size = 0
    
    if "_" in split_by:
        split_by, n_weeks = split_by.split("_")
        n_weeks = int(n_weeks)
        
    val_size = 0.10
    if test:
        test_size = 0.10
    else:
        test_size = 0.0
    
    if split_by == "coursenumber":
        course_numbers = course_dates["coursenumber"].unique()
        indices = np.arange(0, len(course_numbers))
        
        raise ValueError("Implement val and testsize")
        slice_size = int(len(indices) * 0.15)
        
        rng.shuffle(indices)

        val_indices = indices[:slice_size]
        test_indices = indices[slice_size : 2*slice_size]
        train_indices = indices[2*slice_size:]
        
        train_set = course_dates[course_dates["coursenumber"].isin(course_numbers[train_indices])].reset_index(drop=True)
        val_set = course_dates[course_dates["coursenumber"].isin(course_numbers[val_indices])].reset_index(drop=True)
        test_set = course_dates[course_dates["coursenumber"].isin(course_numbers[test_indices])].reset_index(drop=True)
        
    elif split_by == "time":
        weeks = course_dates["calendarweek"].unique()
        train_indices = weeks[:n_weeks]
        val_indices = weeks[:n_weeks+1]
        test_indices = weeks[:n_weeks+2]
        
        train_set = course_dates[course_dates["calendarweek"].isin(train_indices)].reset_index(drop=True)
        val_set = course_dates[course_dates["calendarweek"].isin(val_indices)].reset_index(drop=True)
        test_set = course_dates[course_dates["calendarweek"].isin(test_indices)].reset_index(drop=True)
        
        return (train_set, val_set, test_set), ()
        
    elif split_by == "random":
        
        all_weeks = sorted(course_dates["calendarweek"].unique())
        weeks = all_weeks[:n_weeks]
        
        dataset = course_dates[course_dates["calendarweek"].isin(weeks)].reset_index(drop=True)
        
        indices = np.arange(0, len(dataset))
        rng.shuffle(indices)
        
        val_size_indices = int(len(indices) * val_size)
        val_indices = indices[ :val_size_indices]
        
        test_size_indices = int(len(indices) * test_size)
        test_indices = indices[val_size_indices : val_size_indices + test_size_indices]
        
        train_indices = indices[val_size_indices + test_size_indices:]
        
        datasets = dataset
        indices = (train_indices, val_indices, test_indices)
        
    else:
        raise ValueError("Unknown split criteria.")
    
    #occ_time_series = data_dict[room_id]
    #total_size += len(occ_time_series)
    
    # generate chunks with size 0.05 of the data
    # index_shift = int(len(course_numbers) * 0.05)
    # print(index_shift)


    #if verbose:
    #    print("############## Split Summary ##############")
    #    print("Size of Validationset:", len(val_set)/len(course_dates))
    #    print("Size of Testset:", len(test_set)/len(course_dates))
    #    print("Size of Trainset:", len(train_set)/len(course_dates))
    #    print()
    
    return datasets, indices               
                 
class LectureFeatureEngineer():
    course_features = {"occcount", "occrate", "registered", "exam", "test", "tutorium", 
                       "starttime", "endtime", "calendarweek", "weekday",
                       "roomid","roomcapacity", "type",
                       "studyarea", "ects", "level", "coursenumber"}
    permissible_features = course_features
    
    def __init__(self, data_dict, course_dates_data, course_info_data, dfguru, helpers_path):
        
        self.ts_data_dict = data_dict
        self.helpers_path = helpers_path
        
        min_list = []
        max_list = []
        for room_id in self.ts_data_dict:
            min_timestamp = self.ts_data_dict[room_id]["datetime"].min().replace(hour=0, minute=0, second=0, microsecond=0)
            max_timestamp = self.ts_data_dict[room_id]["datetime"].max().replace(hour=0, minute=0, second=0, microsecond=0) + DateOffset(days=1)
            min_list.append(min_timestamp)
            max_list.append(max_timestamp)
            
        min_timestamp = min(min_list)
        max_timestamp = max(max_list)
        
        self.dfg = dfguru
        self.course_dates_table = dfguru.filter_by_timestamp(
            dataframe = course_dates_data,
            start_time = min_timestamp,
            end_time = max_timestamp,
            time_column = "start_time"
        )

        self.course_info_table = course_info_data  
        self.course_info_table["level"].fillna("None_level", inplace=True)
        self.course_info_table["study_area"].fillna("None_sa", inplace=True)
        self.course_types = ["VL", "UE", "KS"]
        #self.course_types = self.course_info_table["type"].unique()
        # VL: VL, VO, KO
        # UE: UE, AG, IK, PR, PS
        # KS: KS, VU, KV, RE, UV
        # SE: SE
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
            "UV": "KS",}
        
        self.study_areas = self.course_info_table["study_area"].unique()
        self.levels = self.course_info_table["level"].unique()
        
        self.course_numbers = self.course_dates_table["course_number"].unique()

        dictionary = {
            "study_areas": self.study_areas.tolist(),
            "levels": self.levels.tolist(),
            "course_types": self.course_types,
            "course_numbers": self.course_numbers.tolist(),
        }
        # save auxillary data
        with open(file=self.helpers_path, mode="w") as file:
            json.dump(dictionary, file, indent=4)
                      
    def derive_features(self, features):
        
        # get the course number
        # check if features are permissible
        feature_set = set(features)
        set_diff = feature_set.difference(self.permissible_features)
        if set_diff:
            raise ValueError(f"Features {set_diff} are not permissible.")
        
        # check if features are already present
        feature_set_diff = feature_set.difference(self.course_dates_table.columns)

        # include featuers that should always be present
        feature_set_diff = feature_set_diff.union({"coursenumber", "roomid", "starttime", "endtime", "roomcapacity"})
        
        course_dates, feature_set_diff = self.add_features(self.course_dates_table, feature_set_diff)
        
        feature_set = feature_set.union(feature_set_diff)
        
        # remove feature type from feature set
        if "type" in feature_set:
            feature_set.remove("type")
            feature_set = feature_set.union(self.course_types)
            
        # remove feature studyarea from feature set
        if "studyarea" in feature_set:
            feature_set.remove("studyarea")
            feature_set = feature_set.union(self.study_areas)
            
        # remove feature level from feature set
        if "level" in feature_set:
            feature_set.remove("level")
            feature_set = feature_set.union(self.levels)

        with open(file=self.helpers_path, mode="r") as file:
            dictionary = json.load(file)
        dictionary["min_max_registered"] = [str(course_dates["registered"].min()), str(course_dates["registered"].max())]
        with open(file=self.helpers_path, mode="w") as file:
            json.dump(dictionary, file, indent=4)  
        
        return course_dates[sorted(list(feature_set))]

    def add_features(self, course_dates, features):
            
            # initialize all features to 0
            for feature in features:
                course_dates[feature] = 0
                
            if "type" in features:
                course_dates.drop(columns=["type"], inplace=True)
                for course_type in self.course_types:
                    course_dates[course_type] = 0   
                    
            if "studyarea" in features:
                course_dates.drop(columns=["studyarea"], inplace=True)
                for study_area in self.study_areas:
                    course_dates[study_area] = 0
            
            if "level" in features:
                course_dates.drop(columns=["level"], inplace=True)
                for level in self.levels:
                    course_dates[level] = 0
               
            
            for grouping, subdf in course_dates.groupby(["start_time", "end_time", "course_number", "room_id"]):
                
                start_time, end_time, course_number, room_id = grouping
                
                # extract course entry course dates table
                course_dates_mask = (course_dates["start_time"] == start_time) & (course_dates["end_time"] == end_time) & (course_dates["course_number"] == course_number) & (course_dates["room_id"] == room_id)
                
                course_dates_entry = course_dates[course_dates_mask]
                course_info = self.dfg.filter_by_courses(self.course_info_table, [course_number])
                
                if room_id == -1: 
                    
                    # occcount
                    if "occcount" in features:
                        course_dates_entry["occcount"] = -1
                        
                    # occrate
                    if "occrate" in features:
                        course_dates_entry["occrate"] = -1
                            
                else:
                    # extract time series
                    time_series = self.ts_data_dict[room_id]
                    
                    # "occcount", "occrate",            
                    time_series_mask = (time_series["datetime"] >= start_time) & (time_series["datetime"] <= end_time)

                    # occcount
                    if "occcount" in features:
                        course_dates_entry["occcount"] = int(time_series[time_series_mask].max()["CC_estimates"])

                    # occrate 
                    if "occrate" in features:
                        reg_students = course_info["registered_students"].values[0]
                        if reg_students == 0:
                                course_dates_entry["occrate"] = 1
                        else:
                            course_dates_entry["occrate"] = course_dates_entry["occcount"]/reg_students
                        
                        if course_dates_entry["occrate"].values[0] > 10:
                            print(course_dates_entry["occrate"].values[0], course_dates_entry["occcount"].values[0], reg_students)
                            raise
                            course_dates_entry["occrate"] = course_dates["occrate"].max()                    
                    
                if "roomcapacity" in features:
                    course_dates_entry["roomcapacity"] = course_dates_entry["room_capacity"].values[0]
                    
                # registered
                if "registered" in features:
                    course_dates_entry["registered"] = course_info["registered_students"].values[0]
                    
                # starttime
                if "starttime" in features:
                    course_dates_entry["starttime"] = start_time           
                # endtime
                if "endtime" in features:
                    course_dates_entry["endtime"] = end_time
                # calendarweek
                if "calendarweek" in features:
                    course_dates_entry["calendarweek"] = course_dates_entry["calendar_week"].values[0]
                # weekday
                if "weekday" in features:
                    course_dates_entry["weekday"] = course_dates_entry["weekday"].values[0]
                # coursenumber
                if "coursenumber" in features:
                    course_dates_entry["coursenumber"] = course_number
                # roomid
                if "roomid" in features:
                    course_dates_entry["roomid"] = room_id
                # type
                if "type" in features:
                    if len(course_info["type"].unique())  > 1:
                        raise
                    
                    type = self.type_mapping[course_info["type"].unique()[0]]
                    course_dates_entry[type] = 1
                
                if "ects" in features:
                    ects_str = course_info["ects"].values[0]
                    ects = float(".".join(ects_str.split(",")))
                    course_dates_entry["ects"] = ects
                    
                if "studyarea" in features:
                    course_dates_entry[course_info["study_area"].values[0]] = 1

                if "level" in features:
                    course_dates_entry[course_info["level"].values[0]] = 1
                
                course_dates[course_dates_mask] = course_dates_entry
                
                
            course_dates["starttime"] = pd.to_datetime(course_dates["starttime"])
            course_dates["endtime"] = pd.to_datetime(course_dates["endtime"])
            
            if ("starttime" in features) or ("starttime" in course_dates.columns):
                course_dates["time"] = course_dates["starttime"].dt.hour + (course_dates["starttime"].dt.minute / 60)
                course_dates = self.starttime_fourier_series(course_dates, "time")
                course_dates.drop(columns=["time"], inplace=True)
                features = features.union({"starttime1", "starttime2"})
                
            if ("endtime" in features) or ("endtime" in course_dates.columns):
                course_dates["time"] = course_dates["endtime"].dt.hour + (course_dates["endtime"].dt.minute / 60)
                course_dates = self.endtime_fourier_series(course_dates, "time")
                course_dates.drop(columns=["time"], inplace=True)
                features = features.union({"endtime1", "endtime2"})
                
            if ("weekday" in features) or ("starttime" in course_dates.columns):
                course_dates["dow"] = course_dates["starttime"].dt.dayofweek
                course_dates = self.weekday_fourier_series(course_dates, "dow")
                course_dates.drop(columns=["dow"], inplace=True)
                features = features.union({"weekday1", "weekday2"})
                
            if ("calendarweek" in features) or ("starttime" in course_dates.columns):
                course_dates = self.dfg.derive_week(course_dates, "starttime")
                course_dates = self.week_fourier_series(course_dates, "week")
                course_dates.drop(columns=["week"], inplace=True)
                features = features.union({"calendarweek1", "calendarweek2"})
            
            return course_dates, features
        
    def starttime_fourier_series(self, time_series, hourfloat_column):
        time_series["starttime1"] = 0.5 * np.sin(2 * np.pi *  (time_series[hourfloat_column]/24)).round(4) + 0.5
        time_series["starttime2"] = 0.5 * np.cos(2 * np.pi *  (time_series[hourfloat_column]/24)).round(4) + 0.5
        return time_series
    
    def endtime_fourier_series(self, time_series, hourfloat_column):
        time_series["endtime1"] = 0.5 * np.sin(2 * np.pi *  (time_series[hourfloat_column]/24)).round(4) + 0.5
        time_series["endtime2"] = 0.5 * np.cos(2 * np.pi *  (time_series[hourfloat_column]/24)).round(4) + 0.5
        return time_series
        
    def weekday_fourier_series(self, time_series, day_column):
        time_series["weekday1"] = 0.5 * np.sin(2 * np.pi *  (time_series[day_column]/7)).round(2) + 0.5
        time_series["weekday2"] = 0.5 * np.cos(2 * np.pi *  (time_series[day_column]/7)).round(2) + 0.5
        return time_series
    
    def week_fourier_series(self, time_series, week_column):
        time_series["calendarweek1"] = 0.5 * np.sin(2 * np.pi *  (time_series[week_column]/52)).astype(np.float64).round(4) + 0.5
        time_series["calendarweek2"] = 0.5 * np.cos(2 * np.pi *  (time_series[week_column]/52)).astype(np.float64).round(4) + 0.5
        return time_series    
    
class LectureDataset(Dataset):
    
    room_capacities = {0:164, 1:152}
    
    def __init__(self, lecture_df: dict, hyperparameters:dict, dataset_mode:str, validation: bool, path_to_helpers, verbose:bool=True):
        """ Constructor for the occupancy dataset
        Task: Convert the cleaned data into a list of samples
        """
        super().__init__()
        
        self.dataset_mode = dataset_mode

        self.lec_df = lecture_df
        self.hyperparameters = hyperparameters
        self.validation = validation

        ############ Handle Features ############
        self.features_list = hyperparameters["features"].split("_")
        self.features = set(self.features_list)
        
        self.immutable_features = {"type", "studyarea", "ects", "level", "coursenumber"}
        self.immutable_features = self.immutable_features.intersection(self.features)
        
        # derive main feature
        self.occ_feature = self.handle_occ_feature(self.features)
        
        # derive exogenous features
        self.exogenous_features = self.features.difference({"occcount", "occrate"})
        self.exogenous_features = self.exogenous_features.difference(self.immutable_features)
        
        
        # derive exogenous time features
        self.exogenous_features = self.handle_time_features(self.exogenous_features)
          
        # sort features
        self.exogenous_features = sorted(list(self.exogenous_features))


        # derive differencing
        #self.differencing = hyperparameters["differencing"]
        #self.occ_feature, self.exogenous_features, self.sample_differencing = self.handle_differencing_features(self.differencing, self.features, self.occ_feature, self.exogenous_features)
        
        self.path_to_helpers = path_to_helpers
        with open(self.path_to_helpers, "r") as f:
            self.helper = json.load(f)
        
        self.min_registered = int(self.helper["min_max_registered"][0])
        self.max_registered = int(self.helper["min_max_registered"][1])
        
        self.course_numbers = self.helper["course_numbers"]
        self.coursenr_lookup = dict([(x,i) for i,x in enumerate(self.course_numbers)])
        
        if "type" in self.immutable_features:
            self.immutable_features.remove("type")
            self.immutable_features = self.immutable_features.union(self.helper["course_types"])
        
        if "level" in self.immutable_features:
            self.immutable_features.remove("level")
            self.immutable_features = self.immutable_features.union(self.helper["levels"])
            
        if "studyarea" in self.immutable_features:
            self.immutable_features.remove("studyarea")
            self.immutable_features = self.immutable_features.union(self.helper["study_areas"])
        
        
        ############ Derive some helper variables ############
        self.include_x_features = hyperparameters["include_x_features"]
        self.verbose = verbose      
         
        self.discretization = hyperparameters["discretization"] 
        self.occrate_boundaries = torch.arange(hyperparameters["binsize"], 1.0, hyperparameters["binsize"])
        self.onehot = torch.eye(len(self.occrate_boundaries)+1)
        
        ############ Process Data ############
        # convert frequency to timedelta
        #self.td_freq = pd.to_timedelta(hyperparameters["frequency"])
        # process data
        self.samples = self.process_data_df(dataset_mode)
        
        #len_list = [x[1].shape[0] for x in self.samples]
        #print(np.unique(len_list, return_counts=True))
        
        ## correct samples
        #if mode == "normal":
        #    self.samples = self.correct_samples(self.samples, verbose=self.verbose)
        
        #print(f"Len Dataset: {len(self.samples)}")

    ############ Feature Functions ############ 
    def handle_occ_feature(self, features):
            if "occcount" in features:
                return "occcount"
            elif "occrate" in features:
                return "occrate"
            else:
                raise ValueError("No target feature found.")

    def handle_time_features(self, exo_features):
        
        copied_exo_features = exo_features.copy()
        
        if "starttime" in copied_exo_features:
            copied_exo_features.remove("starttime")
            copied_exo_features = copied_exo_features.union({"starttime1", "starttime2"})
            
        if "calendarweek" in copied_exo_features:
            copied_exo_features.remove("calendarweek")
            copied_exo_features = copied_exo_features.union({"calendarweek1", "calendarweek2"})
            
        if "weekday" in copied_exo_features:
            copied_exo_features.remove("weekday")
            copied_exo_features = copied_exo_features.union({"weekday1", "weekday2"})
            
        if "endtime" in copied_exo_features:
            copied_exo_features.remove("endtime")
            copied_exo_features = copied_exo_features.union({"endtime1", "endtime2"})
            
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
    def process_data_df(self, mode):
        
        HS18_count = 0
        HS19_count = 0
        
        list_coursenumbers = []
        for grouping, sub_df in self.lec_df.groupby(["coursenumber"]):

            room_id_vc = np.bincount(sub_df["roomid"]+1, minlength=3)
            ndates = room_id_vc.sum()
            
            if (room_id_vc[1:].sum()/ndates) >= self.hyperparameters["inhall_threshold"]:
                if room_id_vc[1:].sum() >= self.hyperparameters["ndates_threshold"]:
                    list_coursenumbers.append(grouping[0])
                    #print(grouping, room_id_vc, ndates)
                    HS18_count += room_id_vc[1]
                    HS19_count += room_id_vc[2]
        
        self.lec_df = self.lec_df[self.lec_df["coursenumber"].isin(list_coursenumbers)] 
          
        if mode=="time_onedateahead":
            sampling_function = self.create_samples_time_onedateahead
        elif mode=="time_sequential": # standard mode
            sampling_function = self.create_samples_time_sequential
        elif mode == "course_sequential":
            sampling_function = self.create_samples_course_sequential
        else:
            raise ValueError("Unknown mode.")
        
        # handle combined courses
        for grouping, sub_df in self.lec_df.groupby(["starttime","roomid"]):

            if len(sub_df) == 1:
                continue
            else:
                # adapt: occcount, occcrate
                # majority vote: exam, test, tutorium
                if grouping[1] == -1:
                    continue
                
                else:
                    sum_registered = sub_df["registered"].sum()
                    sub_df["occcount"] = (sub_df["occcount"] * sub_df["registered"]/sum_registered).round(0).astype(int)
                    sub_df["occrate"] = sub_df["occcount"] / sub_df["registered"]
                    sub_df[["exam", "test", "tutorium"]] = sub_df[["exam", "test", "tutorium"]].mode().iloc[0]

                    self.lec_df.loc[sub_df.index, ["occcount", "occrate", "exam", "test", "tutorium"]] = sub_df[["occcount", "occrate", "exam", "test", "tutorium"]]

        # handle splitted courses
        for _, sub_df in self.lec_df.groupby(["starttime", "coursenumber"]):
            
            if len(sub_df) > 1:
                # adapt -> registered, occrate                
                sub_df["registered"] = (sub_df["registered"] / len(sub_df)).astype(int)
                
                sub_df["occrate"] = (sub_df["occcount"] / sub_df["registered"])
                
                sub_df["occrate"][sub_df["roomid"]==-1] = -1

                # if room_id -1 and else throw error
                #if (-1 in sub_df["roomid"].values) and (len(sub_df["roomid"].unique()) > 1):
                #    print(sub_df[["occrate", "registered", "roomid"]])
                    
                self.lec_df.loc[sub_df.index, ["registered", "occrate"]] = sub_df[["registered", "occrate"]]


        self.lec_df["registered"]  = (self.lec_df["registered"] - self.min_registered) / (self.max_registered - self.min_registered)
        
        self.min_occrate = self.lec_df["occrate"].min()
        self.max_occrate = self.lec_df["occrate"].max()
        
        # normalize only where occrate is not -1
        mask = self.lec_df["occrate"] != -1
        self.lec_df["occrate"][mask] = (self.lec_df["occrate"][mask] - self.min_occrate) / (self.max_occrate - self.min_occrate)
        
        samples = sampling_function()
        
        return samples

    def create_samples_time_sequential(self):
        
        samples = []
        
        # unique -> type, coursenumber, registered
        #unique_features = self.unique_types
        
        if self.validation:
            max_week = self.lec_df["calendarweek"].max()
        
        for lecture_id, sub_df in self.lec_df.groupby("coursenumber"):
            
            sorted_immu_features = sorted(list(self.immutable_features))
            
            if "coursenumber" in sorted_immu_features:
                sorted_immu_features.remove("coursenumber")
                
            lecture_immutable_features = torch.Tensor(sub_df[sorted_immu_features].iloc[0].values)

            if "coursenumber" in self.immutable_features:
                
                course_nr = sub_df["coursenumber"].iloc[0]
                if type(course_nr) == str:
                    course_nr_string = course_nr
                else:
                    course_nr_string = "{:.3f}".format(sub_df["coursenumber"].iloc[0])
                
                lecture_immutable_features = torch.cat([lecture_immutable_features, torch.Tensor([self.coursenr_lookup[course_nr_string]])])                          

            for i in range(1, len(sub_df)):
                
                X_df = sub_df.iloc[:i]
                y_df = sub_df.iloc[i]
                
                if self.validation:
                    if y_df["calendarweek"] != max_week:
                        continue
                
                X = torch.Tensor(X_df[self.occ_feature].values)
                y = torch.Tensor([y_df[self.occ_feature]])
                
                
                if len(y.shape) == 1:
                    y = y[:, None]
            
                if self.discretization:
                    
                    X = torch.bucketize(X, self.occrate_boundaries)
                    y = torch.bucketize(y, self.occrate_boundaries)
                    
                    X = self.onehot[X]#.squeeze()
                    y = self.onehot[y]#.squeeze()
            
                if self.include_x_features:
                    if len(X.shape) == 1:
                        X = X[:, None]
                    X = torch.cat([X, torch.Tensor(X_df[self.exogenous_features].astype(float).values)], dim=-1)
                
                
                y_features = torch.Tensor(y_df[self.exogenous_features].astype(float).values)[None, :]

                info = (lecture_id, X_df["starttime"], y_df["starttime"], self.exogenous_features, X_df["roomid"], y_df["roomid"], lecture_immutable_features)
                
                if X_df.loc[X_df.index[-1], "starttime"] == y_df["starttime"]:
                    
                    print("Warning: Same Starttime")
                    
                    #print(X_df.loc[X_df.index[-1], "starttime"] , y_df["starttime"])
                    if X_df.loc[X_df.index[-1], "roomid"]  == y_df["roomid"]:
                        print("room_id matches -> nothing to do")
                        print()
                        continue
                    
                    else:
     
                        info_old, X_old, _, _ = samples[-1]
                        (_, X_df_st_old, _, _, X_df_ri_old, _, _) = info_old
                        X = X_old
                        info = (lecture_id, X_df_st_old, y_df["starttime"], self.exogenous_features, X_df_ri_old, y_df["roomid"], lecture_immutable_features)
                        print("-> handled!")
                        print()
                
                if y == -1:
                    continue
                
                samples.append((info, X, y_features, y))

        return samples     
     
      
    def create_samples_time_onedateahead(self):    
        
        samples = []

        if self.validation:
            max_week = self.lec_df["calendarweek"].max()
            
        # group df by lecture
        for lecture_id, sub_df in self.lec_df.groupby("coursenumber"):
            
            sorted_immu_features = sorted(list(self.immutable_features))
            
            if "coursenumber" in sorted_immu_features:
                sorted_immu_features.remove("coursenumber")
                
            lecture_immutable_features = torch.Tensor(sub_df[sorted_immu_features].iloc[0].values)

            if "coursenumber" in self.immutable_features:
            
                course_nr = sub_df["coursenumber"].iloc[0]
                if type(course_nr) == str:
                    course_nr_string = course_nr
                else:
                    course_nr_string = "{:.3f}".format(sub_df["coursenumber"].iloc[0])
                
                lecture_immutable_features = torch.cat([lecture_immutable_features, torch.Tensor([self.coursenr_lookup[course_nr_string]])])                          


            for window in sub_df.rolling(window=2):
                
                if len(window) == 2:

                    X_df = window.iloc[0]
                    y_df = window.iloc[1]
                    
                    if self.validation:
                        if y_df["calendarweek"] != max_week:
                            continue
                        
                    X = torch.Tensor([X_df[self.occ_feature]])
                    y = torch.Tensor([y_df[self.occ_feature]])
                    
                    if self.discretization:
                        
                        X = torch.bucketize(X, self.occrate_boundaries)
                        y = torch.bucketize(y, self.occrate_boundaries)
                        
                        X = self.onehot[X].squeeze()
                        y = self.onehot[y].squeeze()

                    info = (lecture_id, X_df["starttime"], y_df["starttime"], self.exogenous_features, X_df["roomid"], y_df["roomid"], lecture_immutable_features)
                    
                    y_features = torch.Tensor(y_df[self.exogenous_features].astype(float).values)
                                        
                    if self.include_x_features:

                        X = torch.cat([X, torch.Tensor(X_df[self.exogenous_features].astype(float).values)])
                    
                    
                    if X_df["starttime"] == y_df["starttime"]:
                        print()
                        print("Warning: Same Starttime")
                        if X_df["roomid"] == y_df["roomid"]:
                            print("room_id matches -> nothing to do")
                            continue
                        else:
                            info_old, X_old, _, _ = samples[-1]
                            (_, X_df_st_old, _, _, X_df_ri_old, _, _) = info_old
                            X = X_old
                            info = (lecture_id, X_df_st_old, y_df["starttime"], self.exogenous_features, X_df_ri_old, y_df["roomid"], lecture_immutable_features)
                            print("-> handled!")
                            print()
                    
                    samples.append((info, X, y_features, y))
   
        return samples

    def create_samples_course_sequential(self):
        
        samples = []
        
        # unique -> type, coursenumber, registered
        #unique_features = self.unique_types
        
        for lecture_id, sub_df in self.lec_df.groupby("coursenumber"):
            
            lecture_immutable_features = torch.Tensor(sub_df[sorted(list(self.immutable_features))].astype(float).iloc[0].values)
            
            # dropped = sub_df[sorted(list(unique_features))].drop_duplicates()
            # if len(dropped) != 1:
            #     raise ValueError("Unique features are not unique!!")
            
            # unique_part = torch.Tensor(dropped.astype(float).values)
          
            for i in range(1, len(sub_df)):
                
                X_df = sub_df.iloc[:i]
                y_df = sub_df.iloc[i]

                X = torch.Tensor(X_df[self.occ_feature].values)
                y = torch.Tensor([y_df[self.occ_feature]])
                
                if self.discretization:
                    
                    X = torch.bucketize(X, self.occrate_boundaries)
                    y = torch.bucketize(y, self.occrate_boundaries)
                    
                    X = self.onehot[X]#.squeeze()
                    y = self.onehot[y]#.squeeze()
            
                # X_occ = X_df[self.occ_feature]

                # X = torch.Tensor(X_occ.values)
                # y = torch.Tensor([y_df[self.occ_feature]])[:, None]
                if self.include_x_features:
                    X = torch.cat([X, torch.Tensor(X_df[self.exogenous_features].astype(float).values)], dim=1)
                
                #y_features = torch.Tensor(y_df[self.exogenous_features].astype(float).values)
                y_features = torch.Tensor(y_df[self.exogenous_features].astype(float).values)[None, :]

                info = (lecture_id, X_df["starttime"], y_df["starttime"], self.exogenous_features, X_df["roomid"], y_df["roomid"], lecture_immutable_features)
                
                if X_df.loc[X_df.index[-1], "starttime"] == y_df["starttime"]:
                    print()
                    print("Warning: Same Starttime")
                    #print(X_df.loc[X_df.index[-1], "starttime"] , y_df["starttime"])
                    if X_df.loc[X_df.index[-1], "roomid"]  == y_df["roomid"]:
                        print("room_id matches -> nothing to do")
                        continue
                    else:
                        info_old, X_old, _, _ = samples[-1]
                        (_, X_df_st_old, _, _, X_df_ri_old, _, _) = info_old
                        X = X_old
                        info = (lecture_id, X_df_st_old, y_df["starttime"], self.exogenous_features, X_df_ri_old, y_df["roomid"], lecture_immutable_features)
                        print("-> handled!")
                        print()
                
                samples.append((info, X, y_features, y))

        
        return samples

    def create_samples_sequential_new(self):
        
        samples = []
        
        # normalize registered if necessary
        if (self.occ_feature == "occrate"):
            self.lec_df["registered"] = self.lec_df["registered"] / self.lec_df["roomcapacity"]
        
        # unique -> type, coursenumber, registered
        if "type" in self.exogenous_features:
            unique_features = self.unique_types
        
        for lecture_id, sub_df in self.lec_df.groupby("coursenumber"):
            
            dropped = sub_df[sorted(list(unique_features))].drop_duplicates()
            if len(dropped) != 1:
                raise ValueError("Unique features are not unique!!")
            
            unique_part = torch.Tensor(dropped.astype(float).values)
            
            for i in range(1, len(sub_df)):
            
                X_df = sub_df.iloc[:i]
                y_df = sub_df.iloc[i]
                
            
                X_occ = X_df[self.occ_feature]
                
                if X_occ.size == 0:
                    X = torch.Tensor([-1])
                else:
                    X = torch.Tensor(X_df[self.occ_feature].values)
                    X = torch.cat([torch.Tensor([-1]), X])
                    
                y = torch.Tensor([y_df[self.occ_feature]])[None, :]

                #print(X, y)
                if self.include_x_features:
                        x_features = torch.Tensor(X_df[self.exogenous_features].astype(float).values)
                        y_features = torch.Tensor(y_df[self.exogenous_features].astype(float).values)[None, :]
                        exo_features = torch.cat([x_features, y_features])
                        #print(torch.cat([x_features, y_features]).shape)
                        
                        #X = torch.cat([X[:, None], torch.Tensor(X_df[self.exogenous_features].astype(float).values)], dim=1)

                X = torch.cat([X[:, None], exo_features], dim=1)
                
                info = (lecture_id, X_df["starttime"], y_df["starttime"], 
                        self.exogenous_features, X_df["roomid"], y_df["roomid"], unique_part)
                samples.append((info, X, y_features, y))
                
        return samples
    
    ############ Dataset Functions ############
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
       
    def set_samples(self, samples):
        self.samples = samples
        
        
         