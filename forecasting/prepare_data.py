from _dfguru import DataFrameGuru as DFG
from _forecasting import OccFeatureEngineer
from pandas.tseries.offsets import DateOffset
import pandas as pd
from _preprocessing import PLCount

dfg = DFG()

data = dfg.read_data(
    path_repo="../data/cleaned_data", 
    file_name="frequency_data", 
)
data = dfg.clean_signal_data(data)
#min_timestamp = data["datetime"].min().replace(hour=0, minute=0, second=0, microsecond=0)
#max_timestamp = data["datetime"].max().replace(hour=0, minute=0, second=0, microsecond=0) + DateOffset(days=1)
course_dates_data = dfg.read_data(
        path_repo="../data/cleaned_data", 
        file_name="course_dates", 
    )
course_info_data = dfg.read_data(
    path_repo="../data/cleaned_data", 
    file_name="course_info", 
)

############## PLCount ################
frequency = "1h"
plcount = PLCount()

data_dict = {}
for room_id, df in data.groupby("room_id"):
    data_filterd_room = dfg.filter_by_roomid(data, room_id)
    occ_list = plcount.run_on_whole_dataset(data_filterd_room, dfg, frequency)
    data_dict[room_id] = pd.concat(occ_list).drop_duplicates(subset="datetime").reset_index(drop=True)

############## OccFeatureEngineer ################
for room_id in data_dict:
    print(f"Room ID: {room_id}")
    occ_time_series = OccFeatureEngineer(
        data_dict[room_id], 
        course_dates_data, 
        course_info_data, dfg
    ).derive_features(
        features=["exam", "lecture"], 
        room_id=room_id
    )      
    print("--------------------")
    print()
    
    data_dict[room_id] = occ_time_series
    
