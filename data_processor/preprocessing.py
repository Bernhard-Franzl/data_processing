import os
from datetime import datetime
from datetime import time 
import pandas as pd
import matplotlib.pyplot as plt

class Preprocessor:
    
    data_directory = "archive"
    date_format = "%Y-%m-%d"
    last_synchronized = "2024-04-07"
    last_synchronized_dt = datetime.strptime(last_synchronized, date_format)
    raw_data_format = ['Entering', 'Time', 
                       'People_IN', 'People_OUT', 
                       'IN_Support_Count', 'OUT_Support_Count', 
                       'One_Count_1', 'One_Count_2']
    
    def __init__(self, path_to_data, room_to_id, door_to_id):
        self.path_to_data = path_to_data # /home/pi_server
        self.room_to_id = room_to_id
        self.door_to_id = door_to_id
              
    #######  Data Extraction Helper Methods ########
    def get_all_sub_directories(self, path_to_dir):
        sub_dirs = sorted(list(os.walk(path_to_dir))[0][1])
        return sub_dirs
    
    def get_all_sub_files(self, path_to_dir): 
        sub_files = sorted(list(os.walk(path_to_dir))[0][2])
        return sub_files
    
    def filter_directories(self, directories:list):
        filtered_dirs = []
        for x in directories:
            day = datetime.strptime(x.split("_")[2], self.date_format)
            if self.last_synchronized_dt < day:
                filtered_dirs.append(x)
        return filtered_dirs
      
    def get_list_of_data_dirs(self):
        path = os.path.join(self.path_to_data, self.data_directory)
        sub_dirs = self.get_all_sub_directories(path)
        filtered = self.filter_directories(sub_dirs)
        return filtered
     
    #######  Data Extraction Methods ########
    def get_data(self, file_name):
        with open(file_name, "r") as file:
            data = file.readlines()
        return data

    def change_time_format(self, dataframe):
        df = dataframe.copy()
        df["Time"] = df["Time"].apply(lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S %Y"))
        return df
        
    def accumulate_raw_data(self, data_directories):
        
        accumulated_format = self.raw_data_format + ["Room_ID", "Door_ID"]
        df_accumulated = pd.DataFrame(columns=self.raw_data_format)
        samples = 0
        for data_dir_name in data_directories:

            path = os.path.join(self.path_to_data, self.data_directory, data_dir_name)
            file_list = self.get_all_sub_files(path)
            
            # sanity check
            # check if the directory contains the correct files
            #if "door1.csv", "door2.csv", "format.csv":
            if not "door1.csv" in file_list or not "door2.csv" in file_list or not "format.csv" in file_list:
                print(path)
                print(file_list)
                raise ValueError("Data directory does not contain the correct files")
            
            room_name = data_dir_name.split("_")[1]
            room_id = self.room_to_id[room_name]
            
            for x in file_list[:-1]:
                
                door_name = x.split(".")[0]
                door_id = self.door_to_id[door_name]
                
                file_path = os.path.join(path, x) 

                df = pd.read_csv(file_path, names=self.raw_data_format)
                
                df = self.change_time_format(df).sort_values(by="Time", ascending=False)
                df["Room_ID"] = room_id
                df["Door_ID"] = door_id

                samples += len(df)
                df_accumulated = pd.concat([df_accumulated, df], axis=0)
        
        return df_accumulated.reset_index(drop=True)
      
    #######  Data Cleaning Methods ########
    def correct_entering_column(self, entry):
        if entry == "True":
            return 1
        elif entry == "False":
            return 0
        else:
            return int(entry)
        
    def clean_raw_data(self, dataframe):
        
        # make copy of dataframe
        df = dataframe.copy()
        # get nan values
        print(df[df["Entering"].isna()])
        
        # correct the data types
        for col in df.columns[2:]:
            df[col] = df[col].astype(int)
        
        df["event_type"] = df["Entering"].apply(lambda x: self.correct_entering_column(x))
        
        # convert columnnames to lowercase
        df.columns = df.columns.str.lower()
        
        # rename columns
        df = df.rename(columns={"one_count_1":"sensor_one_support_count", 
                                "one_count_2":"sensor_two_support_count"})

        # drop unneccessary columns
        df = df.drop(columns=["entering", "people_in", "people_out"])
        
        # only keep rows with event_type 1 or 0
        df = df[df["event_type"].isin([0,1])].reset_index(drop=True)
        
        # deal with events with low directional support!
        df_test = df.copy()
        
        k = 5
        n = 3
        upper_bound = 3
        
        df_test = df_test[(df_test["in_support_count"] < upper_bound) 
                          & (df_test["out_support_count"] < upper_bound)]

        for x in df_test.index:
            # use index to get row
            rows = df.iloc[x-k:x+k]
            
            # from the neighborhood select the n with the closest time stamp 
            time_diff = rows["time"] - df.loc[x]["time"]
            idx = abs(time_diff).sort_values()[1:n+1].index
            rows_filtered = rows.loc[idx]

            value_counts = rows_filtered["event_type"].value_counts([0,1])
            common_event_type = value_counts.idxmax()
            
            df.loc[x, "event_type"] = common_event_type
        
        # sort by time
        df = df.sort_values(by="time", ascending=True).reset_index(drop=True)
        
        # discard samples betwen 22:00 and 07:30
        lb = time(hour=7, minute=40, second=0)
        ub = time(hour=20, minute=00, second=0)
        mask  = df.apply(lambda x: (x["time"].time() >= lb) & (x["time"].time() <= ub), axis=1)
        df = df[mask].reset_index(drop=True)
        
        return df
    
    ###### Preprocessing Application ########
    def apply_preprocessing(self):
        list_dirs = self.get_list_of_data_dirs()
        data = self.accumulate_raw_data(list_dirs)
        cleaned_data = self.clean_raw_data(data)
        return cleaned_data