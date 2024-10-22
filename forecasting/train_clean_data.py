from _dfguru import DataFrameGuru as DFG
from _preprocessing import PLCount
from _preprocessing import SignalPreprocessor
import json
import pandas as pd
from _plotting import DataPlotter
import os

dfg = DFG()
room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}

harry_plotter = DataPlotter(
    save_path="_plotting/plots/preprocessing/plot", 
    dataframe_guru=dfg,
    plot_height=500,
    plot_width=750
)

############## SignalPreprocessor ################
data_path = "/home/berni/Dropbox/data_29_06_merged/archive"
path_to_json = "_preprocessing/parameters/parameters_best.json"

params = json.load(open(path_to_json, "r"))

preprocessor = SignalPreprocessor(data_path, room_to_id, door_to_id)

cleaned_data, raw_data = preprocessor.apply_preprocessing(params)

corrected_data = preprocessor.correct_25_04_HS19(cleaned_data, "/home/berni/github_repos/data_processing/data/logs_25_04_HS19.txt")

############## PLCount ################
for frequency in ["1h", "5min", "15min", "30min", "1min"]:
    plcount = PLCount()

    data_dict = {}
    for room_id, df in corrected_data.groupby("room_id"):
        
        #data_filterd_room = dfg.filter_by_roomid(corrected_data, room_id)
        occ_list = plcount.run_on_whole_dataset(df, dfg, frequency, params)
        data_dict[room_id] = pd.concat(occ_list).drop_duplicates(subset="datetime").reset_index(drop=True)

        #if frequency == "1min":
        #    # visualize for "1min":
        #    plcount_data = pd.concat(occ_list).drop_duplicates(subset="datetime").reset_index(drop=True)
            
        #    plcount_data["day"] = plcount_data["datetime"].dt.date
        #    raw_data["day"] = raw_data["datetime"].dt.date
        #    corrected_data["day"] = corrected_data["datetime"].dt.date
            
        #    for group, plcount_subdf in plcount_data.groupby(["day"]):
                
        #        date = group[0]
                
        #        raw_data_filtered = dfg.filter_by_date(raw_data, "datetime", date)
        #        corrected_data_filtered = dfg.filter_by_date(corrected_data, "datetime", date)                    
                
        #        raw_data_filtered = dfg.filter_by_roomid(raw_data_filtered, room_id)
        #        corrected_data_filtered = dfg.filter_by_roomid(corrected_data_filtered, room_id)   
                
        #        raw_data_filtered.drop(columns=["day"], inplace=True)
        #        corrected_data_filtered.drop(columns=["day"], inplace=True)        
        #        plcount_data_dropped = plcount_subdf.drop(columns=["day"])          
                
        #        corrected_data_filtered = dfg.calc_occupancy_count(corrected_data_filtered, "datetime", frequency)
        #        raw_data_filtered = dfg.calc_occupancy_count(raw_data_filtered, "datetime", frequency)

        #        harry_plotter.plot_preprocessing(
        #            raw_data = raw_data_filtered, 
        #            processed_data = corrected_data_filtered,
        #            plcount_data = plcount_data_dropped, 
        #            save_bool=True, 
        #            show_bool=True,
        #            combined=False
        #        )
        #        raise
        #        break
    
    repo_path = f"data/freq_{frequency}"
    if not os.path.exists(repo_path):
        os.makedirs(repo_path)
    
    # save data_dict
    for room_id, df in data_dict.items():
        dfg.save_to_csv(df, repo_path, f"room-{room_id}_cleaned_data_29_08")


#raw_data["day"] = raw_data["datetime"].dt.date
#cleaned_data["day"] = cleaned_data["datetime"].dt.date
#for group, raw_sub_df in raw_data.groupby(["day", "room_id"]):

#    raw_sub_df.drop(columns=["day"], inplace=True)
#    cleaned_filtered_data.drop(columns=["day"], inplace=True)
    
#    raw_sub_df = dfg.calc_occupancy_count(raw_sub_df, "datetime", "1min")
#    cleaned_filtered_data = dfg.calc_occupancy_count(cleaned_filtered_data, "datetime", "1min")
#    # occ per min 
#    harry_plotter.plot_preprocessing(
#        raw_sub_df, 
#        cleaned_filtered_data, 
#        save_bool=True, 
#        show_bool=True
#    )

