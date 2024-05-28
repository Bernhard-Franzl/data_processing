import os
from datetime import datetime, time, timedelta
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
    
    def discard_samples(self, dataframe):
        df = dataframe.copy()
        # discard samples betwen 22:00 and 07:30
        lb = time(hour=7, minute=40, second=0)
        ub = time(hour=20, minute=00, second=0)
        mask  = df.apply(lambda x: (x["time"].time() >= lb) & (x["time"].time() <= ub), axis=1)
        df = df[mask].reset_index(drop=True)
        return df

    def df_room_door_dict(self, dataframe):
        df = dataframe.copy()
        room_door_dict = {}
        for room in df["room_id"].unique():
            room_dict = {}
            for door in df["door_id"].unique():
                mask = (df["room_id"] == room) & (df["door_id"] == door)
                if mask.sum() == 0:
                    continue
                else:
                    room_dict[door] = df[mask].reset_index(drop=True)
            room_door_dict[room] = room_dict
        return room_door_dict
  
    def event_type_majority_vote_closest(self, dataframe, refernce_time, n, targte_removed):
        df = dataframe.copy()
        
        if targte_removed:
            idx = abs(df["time"] - refernce_time).sort_values().index[:n]
        else:
            idx = abs(df["time"] - refernce_time).sort_values().index[1:n+1]
            
        filtered = df.loc[idx]
        value_counts = filtered["event_type"].value_counts()
        common_event_type = value_counts.idxmax()
        
        return common_event_type
    
    def get_neighborhood(self, dataframe, x, k):
        df = dataframe.copy()
        i1 = x-k
        if i1 < 0:
            i1 = 0
        i2 = x+k
        if i2 > len(df):
            i2 = len(df)
        rows = df.loc[i1:i2]
        return rows
    
    def filter_data_1(self, dataframe, k, nm, ub):
        df = dataframe.copy()
        
        df = df[df["event_type"].isin([0,1])].sort_values(by="time", ascending=True).reset_index(drop=True)
       
        # neighborhood size
        k = k
        # number of closest neighbors
        n = n
        # upper bound for support count
        upper_bound = ub
         
        dict_df_room_door = self.df_room_door_dict(df)
        df_return = pd.DataFrame(columns=list(df.columns))
        df_return = df_return.astype(df.dtypes)
        
        for room, room_dict in dict_df_room_door.items():
            for door, df_room_door in room_dict.items():  
                
                # deal with events with low directional support! 
                df_test = df_room_door.copy()
            
                # filter out samples with low support count
                df_test = df_test[(df_test["in_support_count"] < upper_bound) 
                                & (df_test["out_support_count"] < upper_bound)]
            
        
                for x in df_test.index:
                    # use index to get row
                    x_row = df_room_door.loc[x]
                    x_time = x_row["time"]
                    #x_door = x_row["door_id"]
                    #x_room = x_row["room_id"]
                    # select neighborhood of sample
                    rows = self.get_neighborhood(df_room_door, x, k)
                    # from the neighborhood select the n with the closest time stamp 
                    common_event_type = self.event_type_majority_vote_closest(rows, x_time, nm, targte_removed=False)
                    df_room_door.loc[x, "event_type"] = common_event_type
                    
                df_return = pd.concat([df_return, df_room_door], axis=0)
            
        return df_return
            
    def filter_data_2(self, dataframe, k, ns, nm, s, ub):
        df = dataframe.copy()
        
        df = df[df["event_type"].isin([0,1])].sort_values(by="time", ascending=True).reset_index(drop=True)
         
        dict_df_room_door = self.df_room_door_dict(df)
        df_return = pd.DataFrame(columns=list(df.columns))
        df_return = df_return.astype(df.dtypes)
        
        for room, room_dict in dict_df_room_door.items():
            for door, df_room_door in room_dict.items():  
                
                # deal with events with low directional support! 
                df_test = df_room_door.copy()
            
                # filter out samples with low support count
                df_test = df_test[(df_test["in_support_count"] < ub) 
                                & (df_test["out_support_count"] < ub)]
            
                #handle the samples with low support count
                for x in df_test.index:
                    # use index to get row
                    x_row = df_room_door.loc[x]
                    x_time = x_row["time"]
            
                    # select neighborhood of sample
                    rows = self.get_neighborhood(df_room_door, x, k)
            
                    # try time filter first -> more reliable
                    x_time_lb = x_time - timedelta(seconds=s)
                    x_time_ub = x_time + timedelta(seconds=s)
                    rows_time_filtered = rows[(rows["time"] >= x_time_lb) & (rows["time"] <= x_time_ub)]
            
                    # if only one sample in time window
                    if len(rows_time_filtered) == 1:
                        # select make majority vote with the n closeste neighbors
                        common_event_type = self.event_type_majority_vote_closest(rows, x_time, nm, targte_removed=False)
                    # if more than one sample in time window     
                    else:
                        # make majority vote with the samples in the time window
                        common_event_type = self.event_type_majority_vote_closest(rows_time_filtered, x_time, ns, targte_removed=False)
                
                    df_room_door.loc[x, "event_type"] = common_event_type
                    
                df_return = pd.concat([df_return, df_room_door], axis=0)

        return df_return

    def handle_event_type_5_6(self, dataframe, k, s, m, ns, nm):
        df = dataframe.copy().reset_index(drop=True)
        mask = ((df["event_type"] == 6) | (df["event_type"] == 5))
        
        for x in df[mask].index:
            
            x_row = df.loc[x]
            x_time = x_row["time"]

            rows = self.get_neighborhood(df, x, k)
            #print(rows)
            
            x_time_lb = x_time - timedelta(seconds=s)
            x_time_ub = x_time + timedelta(seconds=s)
            rows_time_filtered = rows[(rows["time"] >= x_time_lb) & (rows["time"] <= x_time_ub)]
            rows_time_filtered = rows_time_filtered[rows_time_filtered["event_type"].isin([0,1])]
            
            if len(rows_time_filtered) > 0:
                if len(rows_time_filtered) == 1:
                    df.loc[x, "event_type"] = rows_time_filtered["event_type"].values[0]
                else:
                    common_event_type = self.event_type_majority_vote_closest(rows_time_filtered, x_time, ns, targte_removed=True)
                    df.loc[x, "event_type"] = common_event_type
            
            else:
                x_time_lb = x_time - timedelta(minutes=m)
                x_time_ub = x_time + timedelta(minutes=m)
                rows_time_filtered = rows[(rows["time"] >= x_time_lb) & (rows["time"] <= x_time_ub)]
                rows_time_filtered = rows_time_filtered[rows_time_filtered["event_type"].isin([0,1])]
                if len(rows_time_filtered) == 0:
                    # mark as invalid and discard later
                    df.loc[x, "event_type"] = -1
                else:
                    
                    common_event_type = self.event_type_majority_vote_closest(rows_time_filtered, x_time, nm, targte_removed=True)
                    df.loc[x, "event_type"] = common_event_type
                    
        # discard invalid samples
        df = df[df["event_type"] != -1].reset_index(drop=True)   
                
        return df
        
    def filter_data_3(self, dataframe, k, ns, nm, s, ub):
        df = dataframe.copy()
        
        df = df[df["event_type"].isin([0,1,5,6])].sort_values(by="time", ascending=True).reset_index(drop=True)
        
        print("Take care of data: \n 14.05.2024, Event Type 5, HS18 Door1")
        dict_df_room_door = self.df_room_door_dict(df)
        df_return = pd.DataFrame(columns=list(df.columns))
        df_return = df_return.astype(df.dtypes)
        
        for room, room_dict in dict_df_room_door.items():
            for door, df_room_door in room_dict.items():  

                # deal with event type 5 and 6
                df_room_door = self.handle_event_type_5_6(df_room_door,k=5, s=3, m=5, ns=1, nm=5)

                # deal with events with low directional support! 
                df_test = df_room_door.copy()
                
                # filter out samples with low support count
                df_test = df_test[(df_test["in_support_count"] < ub) 
                                & (df_test["out_support_count"] < ub)]
            
                for x in df_test.index:
                    # use index to get row
                    x_row = df_room_door.loc[x]
                    x_time = x_row["time"]

                    # select neighborhood of sample
                    rows = self.get_neighborhood(df_room_door, x, k)
        
                    # try time filter first -> more reliable
                    x_time_lb = x_time - timedelta(seconds=s)
                    x_time_ub = x_time + timedelta(seconds=s)
                    rows_time_filtered = rows[(rows["time"] >= x_time_lb) & (rows["time"] <= x_time_ub)]
            
                    # if only one sample in time window
                    if len(rows_time_filtered) == 1:
                        # select make majority vote with the n closestest neighbors
                        common_event_type = self.event_type_majority_vote_closest(rows, x_time, nm, targte_removed=False)
                    
                    # if more than one sample in time window     
                    else:
                        # make majority vote with the samples in the time window
                        common_event_type = self.event_type_majority_vote_closest(rows_time_filtered, x_time, ns, targte_removed=False)
                        
                    df_room_door.loc[x, "event_type"] = common_event_type
                    
                df_return = pd.concat([df_return, df_room_door], axis=0)

        return df_return
                    
    def clean_raw_data(self, dataframe):
        
        # make copy of dataframe
        df = dataframe.copy()
        # get nan values
        #print(df[df["Entering"].isna()])
        df.dropna(subset=["Entering"], inplace=True)
        #print(df[df["Entering"].isna()])
        
        # drop duplicates
        df.drop_duplicates(inplace=True)
        
        # correct the data types
        for col in df.columns[2:]:
            df[col] = df[col].astype(int)
        
        # delete hidden file in folder data_HS19_2024-04-25
        #print(df[df["Entering"]=="l2"])
        df["event_type"] = df["Entering"].apply(lambda x: self.correct_entering_column(x))
        
        # convert columnnames to lowercase
        df.columns = df.columns.str.lower()
        
        # rename columns
        df = df.rename(columns={"one_count_1":"sensor_one_support_count", 
                                "one_count_2":"sensor_two_support_count"})

        # drop unneccessary columns
        df = df.drop(columns=["entering", "people_in", "people_out"])
        
        # discard samples between 22:00 and 07:30
        df = self.discard_samples(df)
        # without => mse:149.23
        
        #df = self.filter_data_1(df, k=5, n=3, ub=3) # most basic filtering
        #mse: 150.411. k=5, n=3, ub=3
        
        #df = self.filter_data_2(df, k=5, ns=1, nm=5, s=2, ub=3)
        #mse: 145.0588, k=5, ns=1, nm=5, s=2, ub=3
        
        df = self.filter_data_3(df, k=5, ns=1, nm=5, s=2, ub=3)
        
        # sort by time
        df = df.sort_values(by="time", ascending=True).reset_index(drop=True)
        
        return df
    
    ###### Preprocessing Application ########
    def apply_preprocessing(self):
        list_dirs = self.get_list_of_data_dirs()
        data = self.accumulate_raw_data(list_dirs)
        cleaned_data = self.clean_raw_data(data)
        return cleaned_data