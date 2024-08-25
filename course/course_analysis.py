import pandas as pd
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import json
import sys
import numpy as np
sys.path.insert(0, "/home/berni/github_repos/data_processing")
#sys.path.insert(0, "/home/franzl/github_repos/data_processing")

from forcasting.preprocessing.signal_analysis import SignalAnalyzer

class CourseAnalyzer():
    
    room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
    door_to_id = {"door1":0, "door2":1}
    room_capacities = {0:164, 1:152}

    def __init__(self, path_to_courses, path_to_signal):

        # class variables
        self.path_to_courses = path_to_courses
        self.path_to_signal = path_to_signal
        
        # course information dataframe
        self.df_course_info = self.import_from_csv(self.path_to_courses, "course_info", load_dtypes=True)
        # course dates dataframe
        self.df_course_dates = self.import_from_csv(self.path_to_courses, "course_dates", load_dtypes=True)


        # enrich dates with course information
        self.df_combined = self.add_course_info(self.df_course_dates, self.df_course_info)
        self.df_combined.drop("room_id_y", axis=1, inplace=True)
        self.df_combined.rename(columns={"room_id_x":"room_id"}, inplace=True)
        self.df_combined = self.df_combined.drop_duplicates().reset_index(drop=True)
        
        self.signal_analyzer = SignalAnalyzer()
        ## get signal data
        self.df_frequency_data = self.import_from_csv(self.path_to_signal, "frequency_data", load_dtypes=True)
        
        self.signal_params = {
            "prediction_mode":"max",
            "m_before":1,
            "m_after":5,
            "part_mode":"median",
            "max_mean_cutoff":0.3
        }
               
    ###### Basic Methods #####
    def import_from_csv(self, path, file_name, load_dtypes):
        data = pd.read_csv(os.path.join(path, f"{file_name}.csv"), header=0)
        if load_dtypes:
            dtypes = pd.read_csv(os.path.join(path, f"{file_name}_dtypes.csv"), index_col=0)
            data = data.astype(dtypes.to_dict()["0"])
        return data

    def format_dates(self, dataframe_dates):
        df_dates = dataframe_dates.copy(deep=True)
        
        df_dates["start_time"] = df_dates.apply(lambda x: x["Datum"] + " " + x["Startzeit"], axis=1)
        df_dates["start_time"] = df_dates["start_time"].apply(lambda x: datetime.strptime(x, "%d.%m.%y %H:%M"))
        
        df_dates["end_time"] = df_dates.apply(lambda x: x["Datum"] + " " + x["Endzeit"], axis=1)
        df_dates["end_time"] = df_dates["end_time"].apply(lambda x: datetime.strptime(x, "%d.%m.%y %H:%M"))
        
        df_dates.drop(["Datum", "Startzeit", "Endzeit"], axis=1, inplace=True)
        
        return df_dates
    
    def last_entry(self, dataframe, column):
        return dataframe.reset_index().loc[len(dataframe)-1, column]        
    
    def first_entry(self, dataframe, column):
        return dataframe.reset_index().loc[0, column]
       
    def add_course_info(self, dates_dataframe, course_info_dataframe):
        df = dates_dataframe.copy(deep=True)
        df = df.merge(course_info_dataframe, 
                      how="inner", 
                      on="course_number")
        return df
    
    def add_no_dates(self, dates_dataframe):
        
        df = dates_dataframe.copy()
        df["no_dates_dataset"] = df.groupby("course_number")["start_time"].transform("count")        
        
        return df
    
    
    ########## Export/Import Methods ##########
    def export_csv(self, dataframe, path):
        dataframe.to_csv(path, index=False)
    
    def export_metadata(self, path, start_time, end_time, course_numbers):
        start_time = start_time.strftime("%d.%m.%Y %H:%M")
        end_time = end_time.strftime("%d.%m.%Y %H:%M")
        metadata = {"start_time":start_time,
                    "end_time":end_time,
                    "course_numbers":course_numbers,
                    "room_to_id":self.room_to_id, 
                    "door_to_id":self.door_to_id,
                    "room_capacities":self.room_capacities}
        
        with open(path, "w") as file:
            json.dump(metadata, file, indent=4)


    ###### Filter Dataframes ######
    def filter_df_by_timestamp(self, dataframe, start_time, end_time):
        # only show courses betwen start and end time
        df = dataframe.copy(deep=True)
        df = df[(df["start_time"] >= start_time) & (df["end_time"] <= end_time)]
        df = df.sort_values(by="start_time").reset_index(drop=True)
        return df
    
    def filter_df_by_course(self, dataframe, course_number):
        # only show courses betwen start and end time
        df = dataframe.copy(deep=True)
        df = df[df["course_number"] == course_number]
        df = df.sort_values(by="start_time").reset_index(drop=True)
        return df
    
    def filter_df_by_date(self, dataframe, date):
        # only show courses betwen start and end time
        df = dataframe.copy(deep=True)
        mask = df["start_time"].dt.date == date
        df = df[mask]
        df = df.sort_values(by="start_time").reset_index(drop=True)
        return df

    def filter_df_by_room(self, dataframe, room_id):
        # only show courses betwen start and end time
        df = dataframe.copy(deep=True)
        df = df[df["room_id"] == room_id]
        #df = df.sort_values(by="start_time").reset_index(drop=True)
        return df.reset_index(drop=True)
        
    
    ###### Basic Analysis Methods ######
    def handle_combined_courses(self, dataframe):
        df = dataframe.copy(deep=True)
        
        mask = df.duplicated(subset=["start_time", "room_id"], keep=False)
        duplicated = df[mask].drop_duplicates(subset=["start_time", "room_id"], keep="first")
        
        for i, row in duplicated.iterrows():
            start_time = row["start_time"]
            room_id = row["room_id"]
            df_masked = df[(df["start_time"] == start_time) & (df["room_id"] == room_id)]

            df.loc[i, "course_number"] = ", ".join(df_masked["course_number"].sort_values())
            
            df.loc[i, "course_name"] = ",".join(set(df_masked["course_name"]))
            df.loc[i, "max_students"] = df_masked["max_students"].sum()
            df.loc[i, "registered_students"] = df_masked["registered_students"].sum()

        return df.drop_duplicates(subset=["start_time", "room_id"], keep="first").reset_index(drop=True)

    def get_first_last(self, df_first_last, start_time, end_time):
        
        first = False
        last = False
        
        delta = timedelta(hours=1)
        # check if first lecture of the day or no lecture before:
        
        mask_first = ((start_time-delta) < df_first_last["end_time"]) & (df_first_last["end_time"]  < start_time)
        first = not mask_first.any()

        mask_last = ((end_time+delta) > df_first_last["start_time"]) & (df_first_last["start_time"]  > end_time)
        last = not mask_last.any()
        
        return first, last
    
    def calc_course_participants(self, dates_dataframe):
        
        df = dates_dataframe.copy(deep=True)
    
        cur_date = None
        cur_room = None
        
        plot_list = []
        extrema_list = []
        df_list = []

        
        for i,row in tqdm(df.iterrows(), total=len(df)):
            
            if row["course_number"] != "364.000":
                continue
            
            if (cur_date != row["start_time"].date()) or (cur_room != row["room_id"]):
                # we could somehow chache it to avoid recalculating
                cur_date = row["start_time"].date()
                df_first_last = self.filter_df_by_date(self.df_course_dates, cur_date)
                
                cur_room = row["room_id"]
                df_first_last = self.filter_df_by_room(df_first_last, cur_room)
            
            df_signal_cur = self.filter_df_by_room(self.df_frequency_data, cur_room)                
            
            start_time = row["start_time"]
            end_time = row["end_time"]
            
            # check if first or last lecture of the day
            first, last = self.get_first_last(df_first_last, start_time, end_time)
            
            # sanity check df_signal must only contain data in the correct room
            dataframes, participants, extrema, df_list_plotting, _ =  self.signal_analyzer.calc_participants(
                dataframe=df_signal_cur, 
                start_time = start_time, 
                end_time = end_time, 
                first = first, 
                last = last, 
                control=True,
                params=self.signal_params)
            
            if max(participants) == 0:
                diff_ratio = 10
            else:
                diff_ratio = abs(np.diff(participants))/max(participants)
                

            print(i, diff_ratio, participants, row["course_number"], first, last)
            df_help = self.signal_analyzer.merge_participant_dfs(df_list_plotting)
            df_help.to_csv(f"data/df_help_{i}.csv", index=False)
            
            if diff_ratio > self.signal_params["max_mean_cutoff"]:
                part_estimate = int(np.max(participants))
                df.loc[i, "max_or_median"] = "max"
            else:
                part_estimate = int(np.median(participants))
                df.loc[i, "max_or_median"] = "median"
            
            df.loc[i, "present_students_b"] = participants[0]
            df.loc[i, "present_students_a"] = participants[1]
            
            df.loc[i, "present_students"] = part_estimate
            
            df.loc[i, "first"] = first
            df.loc[i, "last"] = last
            
            # description of during dataframe
            df_during = dataframes[1]
            description_during = self.signal_analyzer.describe_inside(df_during)
            
            max_min = description_during["max"] - description_during["min"]
            df.loc[i, "max-min"] = max_min
            df.loc[i, "min_idx"] = df_during["people_inside"].argmin()
            
            min_diff_idx = df_during["people_inside"].diff().argmin()
            df.loc[i, "min_diff_indx"] = df_during["people_inside"].diff().argmin()
            
            duration = end_time - start_time
            duration_min = duration.total_seconds()//60
            df.loc[i, "duration"] = duration_min
            df.loc[i, "overlength"] = duration > timedelta(hours=1, minutes=30)
            
            # before 80% of the time is over, the minimum is reached
            constraint1 = min_diff_idx/duration_min < 0.8  
            # max-min > 0.8 * present_students
            constraint2 = max_min > 0.8 * df.loc[i, "present_students"]
            
            df.loc[i, "irregular"] = (constraint1 & constraint2) | df.loc[i, "overlength"]
            
            plot_list.append(df_list_plotting)
            extrema_list.append(extrema)
            df_list.append(dataframes)
            
        df = self.calc_relative_registered(df)
        df = self.calc_relative_capacity(df)
        
        return df, df_list, extrema_list, plot_list
        
    def calc_relative_registered(self, dataframe):
        # relative = present_students / registered_students
        df = dataframe.copy(deep=True)
        df["relative_registered"] = (df["present_students"] / df["registered_students"]).round(4)
        return df

    def calc_relative_capacity(self, dataframe):
        # relative = present_students / room_capcity
        df = dataframe.copy(deep=True)
        df["relative_capacity"] = (df["present_students"] / df["room_capacity"]).round(4)
        return df
          
                
    ###### Course Attendance Dynamics ######
    def students_running_late(self, dataframe, minutes_interval, minutes_max):
        
        # simply take first n elements of the dataframe
        df_masked = dataframe[minutes_interval-1:minutes_max+1:minutes_interval].reset_index(drop=True)
        df_masked["diff_inside"] = df_masked["people_inside"].diff()
        df_masked.loc[0, "diff_inside"] = df_masked.loc[0, "people_inside"]
        return df_masked
    
    def students_leaving_early(self, dataframe, minutes_interval, minutes_max):
        
        df_masked = dataframe[-minutes_max-1::minutes_interval].reset_index(drop=True)
        cols = ["people_in", "people_out", "people_inside"]
        norm = df_masked.loc[0,cols]
        df_masked[cols] -= norm
        df_masked["diff_inside"] = df_masked["people_inside"].diff()
        return df_masked[1:].reset_index(drop=True)
    
    def process_chunk(self, chunk, periods):
        first_row = chunk.iloc[0]
        start_time = first_row["time"] - timedelta(minutes=periods)
        last_row = chunk.iloc[-1]   
        end_time = last_row["time"]
        #value = abs(chunk["diff_inside"]).max()
        return first_row.name-periods, start_time, last_row.name, end_time
    
    def extract_time_intervals(self, dataframe, periods):
        
        dataframe["diff_minutes"] = dataframe["time"].diff()
        mask = (dataframe["diff_minutes"] >= timedelta(minutes=periods))
        split_points = dataframe[mask].index
        if len(split_points) == 0:
            return []
        # split the dataframe into n parts using the indices in split_points:
        chunks = []
        last_idx = 0
        for idx in split_points:
            chunk = dataframe.loc[last_idx:idx-1]
            chunks.append(self.process_chunk(chunk, periods))
            last_idx = idx
        
        chunk = dataframe.loc[last_idx:]
        chunks.append(self.process_chunk(chunk, periods))
        return chunks
      
    def arbitrary_dynamics(self, dataframe, periods, k):
        
        df = dataframe.copy(deep=True)
        df["diff_inside"] = df["people_inside"].diff(periods=periods)
        df = df.dropna().sort_values(by="diff_inside")
        
        top_k = df[-k:][::-1]
        top_k = top_k[top_k["diff_inside"] > 0].sort_values(by="time")
        
        bot_k = df[:k]
        bot_k = bot_k[bot_k["diff_inside"] < 0].sort_values(by="time")

        students_leaving = self.extract_time_intervals(bot_k, periods)
        students_entering = self.extract_time_intervals(top_k, periods)
        
        return students_leaving, students_entering
     
    def convert_timespan_to_df(self, event_list, dataframe, min_inside):
        
        df = dataframe.copy(deep=True)
        
        df_list = []
        for event in event_list:
            start_idx, start_time, end_idx, end_time = event
            df_cur = df.loc[start_idx:end_idx].copy()
            df_cur["people_inside"] = df_cur["people_inside"] - df_cur["people_inside"].iloc[0]
            
            if abs(df_cur["people_inside"].iloc[0]) >= min_inside:
                df_list.append(df_cur)
            
        return df_list
    
    def calc_attendance_dynamics(self, dataframe, minutes_max):
        # variables for running late and leaving early
        minutes_interval = 5
        # variables for arbitrary dynamics
        periods = 3
        k = 10
        
        df_input = dataframe.copy(deep=True)

        df_coming_late = self.students_running_late(df_input,  minutes_interval, minutes_max)
        df_leaving_early = self.students_leaving_early(df_input,  minutes_interval, minutes_max)
        
        students_leaving, students_entering = self.arbitrary_dynamics(df_input, periods=periods, k=k)
        students_entering = [x for x in students_entering if x[2] >= minutes_max]
        students_leaving = [x for x in students_leaving if x[0] <= (len(df_input) - minutes_max)]
        
        df_list_entering_during = self.convert_timespan_to_df(students_entering, df_input, min_inside=0)
        df_list_leaving_during = self.convert_timespan_to_df(students_leaving, df_input, min_inside=0)
        
        return df_coming_late, df_leaving_early, df_list_entering_during, df_list_leaving_during

    def extract_statistics(self, attendance_dynamics):
        df_coming_late, df_leaving_early, df_list_entering_during, df_list_leaving_during =  attendance_dynamics
       
        late_students = self.last_entry(df_coming_late, "people_inside")
        #late_time = self.last_entry(df_coming_late, "time")
        
        leaving_early_students = self.last_entry(df_leaving_early, "people_out")
        #leaving_time = self.first_entry(df_leaving_early, "time")

        entering_students = [(self.first_entry(x,"time"), self.last_entry(x, "people_inside"), self.last_entry(x, "time")) for x in df_list_entering_during]
        leaving_students = [(self.first_entry(x,"time"), self.last_entry(x, "people_inside"), self.last_entry(x, "time")) for x in df_list_leaving_during]
        
        return late_students, leaving_early_students, entering_students, leaving_students
    
    def calc_dynamics_all_dates(self, dates_dataframe, df_list):
        
        df = dates_dataframe.copy(deep=True)
        
        attendance_dynamics_list = []
        entering_students_list = []
        leaving_students_list = []

        for i, row in dates_dataframe.iterrows():
            
            meta_data = row
            df_during = df_list[i][1]
            attendance_dynamics = self.calc_attendance_dynamics(df_during, minutes_max=15)
            
            late_students, leaving_early_students, entering_students, leaving_students = self.extract_statistics(attendance_dynamics)
            
            attendance_dynamics_list.append(attendance_dynamics)
            
            df.loc[i, "late_students"] = late_students
            df.loc[i, "leaving_early_students"] = leaving_early_students
            
            entering_students_list.append(entering_students)
            leaving_students_list.append(leaving_students)
            
            
        return df, entering_students_list, leaving_students_list, attendance_dynamics_list
            

    #def filter_df_by_courses(self, dataframe, course_numbers):
    #    # only show courses betwen start and end time
    #    df = dataframe.copy(deep=True)
    #    df = df[df["course_number"].isin(course_numbers)]
    #    df = df.sort_values(by="start_time").reset_index(drop=True)
    #    return df