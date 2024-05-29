#########  Imports #########
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


# TODO:

# find a way to do paramter tuning of the preprocessing algorithm

# - incorporate measures from the manual control data
# -> Especially inspect the signals of event type 5 followd by 6 
# -> could be a person entering that is cut of in the middle of the signal



#########  Data Preprocessing #########
cleaned_data = Preprocessor(data_path, room_to_id, door_to_id).apply_preprocessing()
cleaned_data.to_csv("cleaned_data.csv")

#cleaned_data = pd.read_csv("cleaned_data.csv", header=0, index_col=0)
#cleaned_data["time"] = pd.to_datetime(cleaned_data["time"])

#########  Data Analysis #########
# evaluate methods of the SignalAnalyzer class
# manual control data

manual_control = pd.read_csv("zählung.csv")
manual_control["room_id"] = manual_control["room"].map(room_to_id)
mse = 0
for index, row in manual_control.iterrows():
    room_id = row["room_id"]

    start_time_int = row["start_time"]
    end_time_int = row["end_time"]
    
    time_string = row["time"].split(":")
    
    date = row["date"].split(".")
    day = int(date[0])
    month = int(date[1])
    
    control_people_in = row["people_in"]
    
    control_time = dt(2024, month, day, int(time_string[0]), int(time_string[1]), 0)
    start_time = dt(2024, month, day, start_time_int//100, start_time_int%100, 0)
    end_time = dt(2024, month, day, end_time_int//100, end_time_int%100, 0)
    
    first = bool(row["first"])
    last = bool(row["last"])
    
    analyzer = SignalAnalyzer()
    data_analysis = analyzer.filter_by_room(cleaned_data, room_id)
    
    delta = timedelta(minutes=30)
    save_df = analyzer.filter_by_time(data_analysis, start_time-delta, end_time+delta)
    
    
    # m is an extremely important parameter -> the one that is used to calculate the extrema
    df_list, participants, extrema, df_list_plotting, control = analyzer.calc_participants(data_analysis, 
                                            start_time=start_time,
                                            end_time=end_time,
                                            first=first,
                                            last=last,
                                            control=True,
                                            mode="median")
    
    
    df_plotting = analyzer.merge_participant_dfs(df_list_plotting)
    
    horizontal_lines = [(participants[0], "black", " before"),
                        (participants[1], "gold", " after")]

    vertical_lines = [(control_time, "green", "start"),]
    
    title = f"Control:{control_people_in}, Time:{control_time}"    
    analyzer.plot_participants_algo(file_name = f"{control_time}_{room_id}.png",
                             participants=df_plotting,
                             df_list = df_list,
                             control = None,
                             extrema = extrema,
                             horizontal_lines=[],
                             vertical_lines=vertical_lines,
                             title=title)

    control_row = df_plotting[df_plotting["time"] == control_time]
    
    #prediction = control_row["people_inside"].values[0] # 143.812
    #prediction = int(np.mean(participants)) # 45.6875
    #prediction = int(np.max(participants)) # 13.125
    #prediciton = int(np.min(participants))
    
    #term = (control_people_in - prediction)**2
    #term = abs(control_people_in - prediction)
    mse += term
    
    #if mse_term > 10:
    #    print("##################")
    #    print("Time: ", control_time)
    #    print("Room: ", room_id)
    #    print("Participants: ", participants)
    #    print("Control: ", control_people_in)
    #    print("Prediction: ", prediction)
    #    print("MSE: ", mse_term)
    #    print(analyzer.describe_inside(df_list_plotting[1])["std"])
    #    print("##################")
    #    print()

print("MSE: ", mse/len(manual_control))
    
#room_id = 0
#door_id = 0

#year = 2024
#month = 4
##day
#start_time_int = 1530
#end_time_int = 1700
#first, last = False, False
#day_list = [8]

#for day in  day_list:

#    print(f"#################### {day}.4.2024 {start_time_int}####################")
#    start_time = dt(year, month, day, start_time_int//100, start_time_int%100, 0)
#    end_time = dt(year, month, day, end_time_int//100, end_time_int%100, 0)
    
#    # set first true if:
#    # - first lecture of the day
#    # - no lecture before
#    # - early termination of the last lecture


#    analyzer = SignalAnalyzer()
#    data_analysis = analyzer.filter_by_room(cleaned_data, room_id)

#    # m is an extremely important parameter -> the one that is used to calculate the extrema
#    df_list, participants, extrema, df_list_plotting, control = analyzer.calc_participants(data_analysis, 
#                                            start_time=start_time,
#                                            end_time=end_time,
#                                            first=first,
#                                            last=last,
#                                            control=True)
    
#    #print(participants)
#    #print(participants[0]-participants[1])
#    #print()
#    print("Participants: ", participants)
    
    #df_during = df_list[1]
    #print(analyzer.describe_inside(df_during))
    
    #min_index = df_during["people_inside"].diff().argmin()
    #print(df_during.iloc[min_index])
    #print(df_during["people_inside"].min() df_during["people_inside"].max())
    #print()
    #df_before, df_during, df_after = df_list

    ## if high std -> check for outliers, for example courses that end very early!
    ## nice to detect irregularities in the data
    #description_during = analyzer.describe_inside(df_during)
    #print(description_during)

#########  Data Visualization #########
#visard = Visualizer()


#df_plotting = visard.merge_participant_dfs(df_plot_list)

#horizontal_lines = [(participants[0], "black", " before"),
#                    (participants[1], "gold", " after")]


#visard.plot_participants(save_path = f"plots/{start_time}.png",
#                         participants=df_plotting,
#                         df_list = df_list,
#                         control = None,
#                         extrema = extrema,
#                         horizontal_lines=[])


# TODO:
# - during the calculation of participants construct a signal that shows the participants over time
# we need that for nice viszalization

# - make a filter that checks for unplausible signal:
# E.g: if one person enters and one leaves after 1 second in the same door 
# -> change direction if necessary, maybe in form of a sliding window

# - make viszalization interactive (plotly)

# - incorporate measures from the manual control data
# -> Especially inspect the signals of event type 5 followd by 6 
# -> could be a person entering that is cut of in the middle of the signal

# - restructure data cleaning methods, restructure api

# DONE:
# - from signal calculate something like: "participants" 
# We need one number representing the participants of course(defined time span)
# Good parameter for m: 3