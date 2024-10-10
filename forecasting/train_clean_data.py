from _dfguru import DataFrameGuru as DFG
from _preprocessing import PLCount
from _preprocessing import SignalPreprocessor
import json
import pandas as pd

dfg = DFG()
room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}

############## SignalPreprocessor ################
data_path = "/home/berni/Dropbox/data_29_06_merged/archive"
path_to_json = "_preprocessing/parameters/parameters_best.json"

params = json.load(open(path_to_json, "r"))

preprocessor = SignalPreprocessor(data_path, room_to_id, door_to_id)

cleaned_data, raw_data = preprocessor.apply_preprocessing(params)

corrected_data = preprocessor.correct_25_04_HS19(cleaned_data, "/home/berni/github_repos/data_processing/data/logs_25_04_HS19.txt")

# create plots that shows the effect of the preprocessing

#harry plotter

raw_data["day"] = raw_data["datetime"].dt.date
cleaned_data["day"] = cleaned_data["datetime"].dt.date
for group, raw_sub_df in raw_data.groupby(["day", "room_id"]):
    
    date, room_id = group
    cleaned_filtered_data = dfg.filter_by_date(cleaned_data, "datetime", date)
    cleaned_filtered_data = dfg.filter_by_roomid(cleaned_filtered_data, room_id)
    
    # occ per min
    
    print(cleaned_filtered_data)
    print(raw_sub_df)
    raise

############## PLCount ################
raise
for frequency in ["1min", "5min", "15min", "30min", "1h"]:
    plcount = PLCount()

    data_dict = {}
    for room_id, df in corrected_data.groupby("room_id"):
        
        #data_filterd_room = dfg.filter_by_roomid(corrected_data, room_id)
        occ_list = plcount.run_on_whole_dataset(df, dfg, frequency, params)
        data_dict[room_id] = pd.concat(occ_list).drop_duplicates(subset="datetime").reset_index(drop=True)
     
    # visualize for "1min":
    if frequency == "1min":
        # harry plotter
        raise
       
    # save data_dict
    for room_id, df in data_dict.items():
        dfg.save_to_csv(df, "data", f"room-{room_id}_freq-{frequency}_cleaned_data_29_08")

