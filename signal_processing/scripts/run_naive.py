from preprocessing import Preprocessor
from signal_analysis import SignalAnalyzer
from datetime import datetime as dt
from datetime import time, timedelta
import pandas as pd
import numpy as np
#########  Constants #########
room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}
data_path = "/home/berni/data_05_26"

########  Data Preprocessing #########
#cleaned_data = Preprocessor(data_path, room_to_id, door_to_id).apply_preprocessing()
#cleaned_data.to_csv("data/cleaned_data.csv")

cleaned_data = pd.read_csv("data/cleaned_data.csv", header=0, index_col=0)
cleaned_data["time"] = pd.to_datetime(cleaned_data["time"])


room_id = 0
year = 2024
month = 4
#day
start_time_int = 1015
end_time_int = 1145
first, last = False, False
day_list = [11]

for day in  day_list:

    print(f"#################### {day}.4.2024 {start_time_int}####################")
    start_time = dt(year, month, day, start_time_int//100, start_time_int%100, 0)
    end_time = dt(year, month, day, end_time_int//100, end_time_int%100, 0)
    
    # set first true if:
    # - first lecture of the day
    # - no lecture before
    # - early termination of the last lecture

    analyzer = SignalAnalyzer()
    data_analysis = analyzer.filter_by_room(cleaned_data, room_id)

    # m is an extremely important parameter -> the one that is used to calculate the extrema
    df_list, participants, extrema, df_list_plotting, control = analyzer.calc_participants(data_analysis, 
                                            start_time=start_time,
                                            end_time=end_time,
                                            first=first,
                                            last=last,
                                            control=True)
    
    df_plotting = analyzer.merge_participant_dfs(df_list_plotting)

 
    analyzer.plot_participants_algo(file_name = f"{day}.{month}.{year}_{start_time_int}_algo.png",
                                participants=df_plotting,
                                df_list = None,
                                control = None,
                                extrema = None,
                                horizontal_lines=[],
                                vertical_lines=[],#vertical_lines,
                                title="Algorithm Approach")
    #print(participants)
    #print(participants[0]-participants[1])
    #print()
    print("Participants: ", participants)
    
    df_during = df_list[1]
    print(analyzer.describe_inside(df_during))
    
    min_index = df_during["people_inside"].diff().argmin()
    print(df_during.iloc[min_index])
    print(df_during["people_inside"].min(), df_during["people_inside"].max())
    print()
    df_before, df_during, df_after = df_list

    # if high std -> check for outliers, for example courses that end very early!
    # nice to detect irregularities in the data
    description_during = analyzer.describe_inside(df_during)
    print(description_during)