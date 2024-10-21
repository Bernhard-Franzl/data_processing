import numpy as np
import os
import random
import torch

from _forecasting import prepare_data, load_data_dicts, prepare_data_lecture, load_data_lecture
from _dfguru import DataFrameGuru as DFG
from _forecasting import OccFeatureEngineer

import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go


dfguru = DFG()
np_rng = np.random.default_rng(seed=42)
torch_rng = torch.Generator()
torch_rng.manual_seed(42)
random.seed(42)


def load_data(path_to_data_dir, frequency, dfguru):
    course_dates_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_dates")

    course_info_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_info")
    
    course_info_data.drop(columns=["room_id"], inplace=True)
    course_info_data.drop_duplicates(inplace=True)
    
    data_dict = {}
    path_to_occ_data = os.path.join(path_to_data_dir, f"freq_{frequency}")
    for room_id in [0, 1]:
        
        ########## Load Data ##########
        occ_time_series = dfguru.load_dataframe(
            path_repo=path_to_occ_data, 
            file_name=f"room-{room_id}_cleaned_data_29_08", 
        )[:-1]
            
        ########## OccFeatureEngineer ##########
        #occ_time_series = OccFeatureEngineer(
        #    occ_time_series, 
        #    course_dates_data, 
        #    course_info_data, 
        #    dfguru,
        #    frequency
        #).derive_features(
        #    features=feature_list, 
        #    room_id=room_id
        #)
            
        data_dict[room_id] = occ_time_series
        
    return data_dict, course_dates_data, course_info_data
        




frequency = "1min"
path_to_data_dir = "data"

data_dict, course_dates_data, course_info_data = load_data(path_to_data_dir, frequency, dfguru)
    
    
# filter by room_id 
room_id = 0
occ_time_series = data_dict[room_id]
print(occ_time_series.head())


raise
print(occ_time_series.head())

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=occ_time_series["datetime"],
        y=np.concatenate(dict_preds[room_id]),
        mode="lines+markers",
        name=f"Prediction Room {room_id}"
    )
)
fig.add_trace(
    go.Scatter(
        x=np.concatenate(dict_y_times[room_id]),
        y=np.concatenate(dict_targets[room_id]),
        mode="lines+markers",
        name=f"Target Room {room_id}"
    )
)

target_feature_names = infos[0][3]
target_features_room = dict_target_features[room_id]
x=np.concatenate(dict_y_times[room_id])
y=np.concatenate(target_features_room) 

for i in range(len(target_feature_names)):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y[:, i],
            mode="lines+markers",
            name=f"{target_feature_names[i]} Room {room_id}"
        )
    )


fig.update_layout(
    title=f"Run {n_run} - Combination {n_comb} - Room {room_id}",
    xaxis_title="Time",
    yaxis_title="Occupancy"
)

fig.show()