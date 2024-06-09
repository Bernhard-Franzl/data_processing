import pandas as pd
from datetime import datetime, time, timedelta 
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt

class SignalAnalyzer():
    def __init__(self):
        pass

    #######  Data Filtering Methods ########
    def filter_by_room(self, dataframe, room_id):
        df = dataframe.copy()
        mask = df["room_id"] == room_id
        return df[mask].reset_index(drop=True)
    
    def filter_by_door(self, dataframe, door_id):
        df = dataframe.copy()
        mask = df["door_id"] == door_id
        return df[mask].reset_index(drop=True)
    
    def filter_by_time(self, dataframe, start_time=None, end_time=None, reset_index=True):
        
        # if no start time and end time is given throw exception
        if (start_time == None) and (end_time == None):
            raise Exception("No start and end time given")
        # if no start time given, set it to the minimum time in the dataframe
        if start_time == None:
            start_time = dataframe["time"].min()
        #if no end time given, set it to the maximum time in the dataframe
        if end_time == None:
            end_time = dataframe["time"].max()
            
        df = dataframe.copy()
        mask = (df["time"] >= start_time) & (df["time"] <= end_time)
        if reset_index:
            return df[mask].reset_index(drop=True)
        else:
            return df[mask]
    
    
    ######## Statistical Methods ########
    def calc_mean_inside(self, dataframe):
        df = dataframe.copy()
        return df["people_inside"].mean()
    
    def calc_median_inside(self, dataframe):
        df = dataframe.copy()
        return df["people_inside"].median()
    
    def calc_mode_inside(self, dataframe):
        df = dataframe.copy()
        return int(df["people_inside"].mode())
    
    def calc_std_inside(self, dataframe):
        df = dataframe.copy()
        return df["people_inside"].std()
    
    def describe_inside(self, dataframe):
        df = dataframe.copy()
        return df["people_inside"].describe()
    
    
    ########  Data Analysis Methods ########
    # calc people entering and leaving
    def calc_entering_leaving(self, dataframe):
        df = dataframe.copy()
        
        # cum sum of event type for people in
        df["people_in"] = df["event_type"].cumsum()
        # cum sum of inverted event type for people out
        df["people_out"] = df["event_type"].apply(lambda x: 1-x).cumsum()
        
        return df

    def calc_inside(self, dataframe):
        # make a copy of the dataframe
        df = dataframe.copy()
        # calc people entering and leaving
        df = self.calc_entering_leaving(df)
        # calc people in room a a given time
        df["people_inside"] = df["people_in"] - df["people_out"]
        return df
    
    def calc_inside_per_min(self, dataframe, n=1, start=None, end=None):
        
        df = dataframe.copy()

        df["people_in"] = df["event_type"].apply(lambda x: 1 if x == 1 else 0)
        df["people_out"] = df["event_type"].apply(lambda x: 1 if x == 0 else 0)
        
        idx = pd.date_range(start=start, end=end, freq=f'{n}min')
        
        df = df.set_index("time")\
                .resample(f"{n}min")\
                .sum().reindex(idx, fill_value=0).reset_index()

        df.rename(columns={"index":"time"}, inplace=True)
        
        df["people_in"] = df["people_in"].cumsum()
        df["people_out"] = df["people_out"].cumsum()
        df["people_inside"] = df["people_in"] - df["people_out"]
        
        df.drop(columns=["in_support_count", "out_support_count", 
                         "sensor_one_support_count", "sensor_two_support_count",
                         "door_id", "event_type", "room_id"], 
                inplace=True)
        return df
            
    def get_time(self, start_time, end_time, first, last):
                
        if first:
            start_time_new = start_time - timedelta(hours=1)
        else:
            start_time_new = start_time - timedelta(minutes=15)
            
        if last:
            end_time_new = end_time + timedelta(hours=1)
        else:
            end_time_new = end_time + timedelta(minutes=15)
            
        return start_time_new, end_time_new
    
    def get_local_extrema(self, dataframe, n):
        df = dataframe.copy()

        df['min'] = df.iloc[argrelextrema(df.people_inside.values, 
                                          np.less_equal,
                                          order=n)[0]]["people_inside"]
        df['max'] = df.iloc[argrelextrema(df.people_inside.values,
                                            np.greater_equal,
                                            order=n)[0]]["people_inside"]
        return df
    
    def get_people_inside(self, dataframe):
        return dataframe.iloc[-1]["people_inside"]
    
    def get_people_in(self, dataframe):
        return dataframe.iloc[-1]["people_in"]
    
    def get_people_out(self, dataframe):
        return dataframe.iloc[-1]["people_out"]
        
    def calc_participants(self, dataframe, start_time, end_time, first, last, control, mode):
        
        #return_tuple = ()
        
        n = 1
        # make a copy of the dataframe
        df = dataframe.copy()
        
        # get the new start and end time
        df_during = self.filter_by_time(df, start_time, end_time)
        
        df_during = self.calc_inside_per_min(df_during, n ,start_time, end_time-timedelta(minutes=1))
        
        start_time_new, end_time_new = self.get_time(start_time, end_time, first, last)
        
        df_before = self.filter_by_time(df, start_time_new, start_time)
        df_before = self.calc_inside_per_min(df_before, n ,start_time_new, start_time-timedelta(minutes=1))
        
        df_after = self.filter_by_time(df, end_time, end_time_new)
        df_after = self.calc_inside_per_min(df_after, n ,end_time, end_time_new-timedelta(minutes=1))
        
        #return_tuple += ([df_before, df_during, df_after],)
        
        if control:
            df_control = self.filter_by_time(df, start_time_new, end_time_new)
            df_control = self.calc_inside_per_min(df_control, n ,start_time_new, end_time_new-timedelta(minutes=1))
            #return_tuple += (df_control,)

        def process_part(dataframe, n=10, before=True, first=False, last=False):
            df = dataframe.copy()
            
            extrema = self.get_local_extrema(dataframe=df, n=n)
            minima_mask = extrema["min"].notnull()
            maxima_mask = extrema["max"].notnull()
            
            min = extrema[minima_mask]
            max = extrema[maxima_mask]
            
            # differentiate between before and after
            # before take first global min
            # after take last global min
            inside_col = min["people_inside"]
            global_min_indices  = min.index
            #global_min_indices = min[inside_col == inside_col.min()].index
            
            if before:
                if first:
                    glob_min_idx = global_min_indices[0]
                else: 
                    glob_min_idx = global_min_indices[-1]
                    
            else:
                if last:
                    glob_min_idx = global_min_indices[-1]
                else:
                    glob_min_idx = global_min_indices[0]

            glob_min = min.loc[glob_min_idx]
            glob_min_time = glob_min["time"]
            
            counter=0
            before_min = df[df["time"] <= glob_min_time]
            after_min = df[df["time"] >= glob_min_time]
            
            df_list = []

            if before:
                if len(before_min) > 1:
                    counter += before_min.iloc[-1]["people_in"]
                    df_list.append((before_min, "before_in"))

                    if len(after_min) > 1:
                        
                        #before_min_in = before_min.iloc[-1]["people_in"]
                        #after_min_in = after_min.iloc[0]["people_in"]
                        inside_col = after_min["people_inside"]
                        counter += inside_col.iloc[-1] - inside_col.iloc[0] #+ (after_min_in -before_min_in)
                        df_list.append((after_min, "before_inside"))

                else:
                    if len(after_min) > 1:
                        inside_col = after_min["people_inside"]
                        counter += inside_col.iloc[-1]
                        df_list.append((after_min, "before_inside"))
                        
                    else:
                        raise Exception("Both dataframes are empty")
                             
            else:
                if len(before_min) > 1:
        
                    last_row_before = before_min.iloc[-1]
                    counter -= last_row_before["people_inside"]
                    df_list.append((before_min, "after_inside"))
                    
                    if len(after_min) > 1:
                        
                        outside_col = after_min["people_out"]        
                        #before_min_out = last_row_before["people_out"]
                        counter += outside_col.iloc[-1] - outside_col.iloc[0] #- before_min_out
                        df_list.append((after_min, "after_out"))
                    
                else:
                    if len(after_min) > 1:
                        outside_col = after_min["people_out"]
                        
                        counter += outside_col.iloc[-1]
                        df_list.append((after_min, "after_out"))
                        
                    else:
                        raise Exception("Both dataframes are empty")
                
            return extrema, counter, df_list
        
        m = 2
        ext_b, in_b, df_list_b = process_part(df_before, n=m, before=True, first=first, last=last)
        ext_a, out_a, df_list_a = process_part(df_after, n=m, before=False, first=first, last=last)
        
        def calc_participants(during:pd.DataFrame, in_before:int, out_after:int, mode:str):
            if mode == "median":
                inside_during = self.calc_median_inside(during)
            elif mode == "mean":
                inside_during = self.calc_mean_inside(during)
            elif mode == "max":
                inside_during = during["people_inside"].max()
            else:
                raise Exception("Mode not implemented")
            
            part_before = in_before + inside_during 
            part_after = out_after - during.iloc[-1]["people_inside"] + inside_during
            sanity_check = abs(part_before - part_after)
            
            return part_before, part_after, sanity_check
            
        

        part_b, part_a, sanity_check = calc_participants(during=df_during, in_before=in_b, 
                                                         out_after=out_a, mode=mode)
        extrema = pd.concat([ext_b, ext_a])
        
        if control:
            return [df_before, df_during, df_after], (part_b, part_a), extrema, [df_list_b, df_during, df_list_a], df_control
        else:
            return [df_before, df_during, df_after], (part_b, part_a), extrema, [df_list_b, df_during, df_list_a]
        
    def calc_participants_simple(self, dataframe, start_time, end_time):
        df = dataframe.copy()
        
        start_time_new = datetime(start_time.year, start_time.month, start_time.day, 7, 45, 0)
        
        df = self.filter_by_time(df, start_time_new, end_time)
        df = self.calc_inside(df)
        
        return df
 
############ Matplotlib ############
    ##### Plot Particpants Algorithm - How it works #####  
    def merge_participant_dfs(self, dataframes:list):
        
        if len(dataframes) != 3:
            raise ValueError("Expected 3 dataframes, got: ", len(dataframes))
        else:
            
            def correct_dataframes(dataframes, cur_last):
                # if dataframes is list
                df_list = []
                if type(dataframes) == list:
                    for df, info in dataframes:
                        pos, col = info.split("_")
                        
                        if col == "out":
                            df["people_out"] = df["people_out"] - df.iloc[0]["people_out"]
                            df["people_inside"] =  (-df["people_out"]) + cur_last
                            cur_last = cur_last + df.iloc[-1]["people_inside"]
                            df_list.append(df)
                            
                            
                        elif col == "in":
                            if cur_last != 0:
                                raise ValueError("First dataframe should start with 0 people inside")
                            
                            df["people_inside"] = df["people_in"] + cur_last
                            cur_last =  df.iloc[-1]["people_inside"]
                            df_list.append(df)

                            
                        else:
                            if pos == "before":
                                df["people_inside"] = df["people_inside"] - df.iloc[0]["people_inside"] + cur_last
                                cur_last = df.iloc[-1]["people_inside"]
                                df_list.append(df)
                                
                            if pos == "after":
                                df["people_inside"] = df["people_inside"] + cur_last
                                cur_last = df.iloc[-1]["people_inside"]
                                df_list.append(df)
            
                    return df_list, cur_last
                                
                else:
                    dataframes["people_inside"] = dataframes["people_inside"] + cur_last
                    cur_last = dataframes.iloc[-1]["people_inside"]
                    return [dataframes], cur_last
            
            df_list = []
            cur_last = 0
            for x in dataframes:        
                dfs, cur_last = correct_dataframes(x, cur_last)
                df_list += dfs

        return pd.concat(df_list).reset_index(drop=True)
    
    def plot_participants_algo(self, file_name, participants=None, 
                        df_list=None, control=None, 
                        extrema=None, 
                        horizontal_lines=[], vertical_lines=[],
                        title=None):
        path = "plots/"
        if (participants is None) and (df_list is None) and (control is None):
            raise ValueError("Either participants, df_list or control must be provided")
        
        else:
            legend = []
            fig, ax = plt.subplots(figsize=(10,5))
            
            if participants is not None:
                participants.plot(x="time", y="people_inside", kind="line", ax=ax)
                legend.append("participants")
            
            if df_list is not None:
                for df in df_list:
                    df.plot(x="time", y="people_inside", kind="line", ax=ax)
                    legend.append("control")
            
            if control is not None:
                control.plot(x="time", y="people_inside", kind="line", ax=ax)
                legend.append("simple")
                
            if extrema is not None:
                extrema.plot(x="time", y="min", kind="scatter", ax=ax, c="r")
                legend.append("min")
                extrema.plot(x="time", y="max", kind="scatter", ax=ax, c="g")
                legend.append("max")
            
            
            if len(horizontal_lines) > 0:
                for x, c, l in horizontal_lines:
                    plt.axhline(y = x, c=c, linestyle = '-')
                    legend.append(f"{l}")
            if len(vertical_lines) > 0:
                for x, c, l in vertical_lines:
                    plt.axvline(x = x, c=c, linestyle = '-')
                    legend.append(f"{l}")
                    
            if title is not None:
                plt.title(title)
            
            plt.xlabel("Time")
            plt.ylabel("People Inside")   
            plt.legend(legend)
            plt.grid()
            fig.savefig(path + file_name)
            
            plt.close()

