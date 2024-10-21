from dash import html, dcc, callback, register_page, Output, Input, State
import pandas as pd
#from components import plot_header, course_info
#from data_handling.data_handler import DataHandler
#from visualization.visualization import Visualizer
from dash import dash_table

import os
from _dfguru import DataFrameGuru as DFG
from _forecasting import OccFeatureEngineer

import plotly.graph_objects as go

dfguru = DFG()
global_feature_list = sorted(list(OccFeatureEngineer.permissible_features))

def load_data(path_to_data_dir, frequency, feature_list, dfguru, room_id):
    course_dates_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_dates")

    course_info_data = dfguru.load_dataframe(
        path_repo=path_to_data_dir, 
        file_name="course_info")
    
    course_info_data.drop(columns=["room_id"], inplace=True)
    course_info_data.drop_duplicates(inplace=True)
    
    path_to_occ_data = os.path.join(path_to_data_dir, f"freq_{frequency}")

    ########## Load Data ##########
    occ_time_series = dfguru.load_dataframe(
        path_repo=path_to_occ_data, 
        file_name=f"room-{room_id}_cleaned_data_29_08", 
    )[:-1]
        
    return occ_time_series, course_dates_data, course_info_data
        

register_page(__name__, path="/")


frequency = "1h"
path_to_data_dir = "data"
data_dict, course_dates_data, course_info_data = load_data(path_to_data_dir, frequency, global_feature_list, dfguru, 0)

# Layout
def layout():
    return html.Div(
        [
            html.H1(''),
            # Dropdown for frequency resampling
            dcc.Dropdown(
                id="ts_viewer_frequency",
                options=[{'label': '1 minute', 'value': '1min'},
                         {'label': '5 minutes', 'value': '5min'},
                         {'label': '15 minutes', 'value': '15min'},
                         {'label': '30 minutes', 'value': '30min'},
                         {'label': '1 hour', 'value': '1h'}],
                value="15min",
                clearable=False,
                style={'width': '50%'}
            ),
            # Dropdown for room filter
            dcc.Dropdown(
                id="ts_viewer_room_filter",
                options=[{'label': f'Room {i}', 'value': i} for i in range(2)],  # Room ids 0 to 5
                value=0,  # Default value
                clearable=False,
                style={'width': '50%'}
            ),
            # Hidden store for keeping the x-axis range
            dcc.Store(
                id="xaxis_range_store"
            ),
            # Graph for time series viewer
            dcc.Graph(
                id="ts_viewer_figure",
            ),
            # DataTable to show filtered course dates
            dash_table.DataTable(
                id='course_dates_table',
                columns=[{"name": i, "id": i} for i in course_dates_data.columns],
                data=[],  # Start with empty data
                page_size=10,  # Adjust the page size as needed
            ),
        ]
    )

@callback(
    Output("ts_viewer_figure", "figure"),
    Output("xaxis_range_store", "data"),
    Input("ts_viewer_room_filter", "value"),
    Input("ts_viewer_frequency", "value"),
    Input("ts_viewer_figure", "relayoutData"),
    State("xaxis_range_store", "data")
)
def update_ts_viewer(room_dropdown_value, 
                     frequency_dropdown_value, 
                     relayout_data, 
                     stored_xaxis_range):

    occ_time_series, course_dates_data, course_info_data = load_data(
        path_to_data_dir, 
        frequency_dropdown_value, 
        global_feature_list, 
        dfguru, 
        room_dropdown_value
    )
    
    # Update the x-axis range if relayoutData has a new range
    if relayout_data and ('xaxis.range' in relayout_data):
        min_time = relayout_data['xaxis.range'][0]
        # convert to datetime
        min_time = pd.to_datetime(relayout_data['xaxis.range'][0])
        max_time = pd.to_datetime(relayout_data['xaxis.range'][1])
        xaxis_range = [min_time, max_time]

    elif relayout_data and ("xaxis.range[0]" in relayout_data):
        # If the relayoutData contains the range in a different format
        # convert to datetime
        min_time = pd.to_datetime(relayout_data['xaxis.range[0]'])
        max_time = pd.to_datetime(relayout_data['xaxis.range[1]'])
        xaxis_range = [min_time, max_time]
    
    elif stored_xaxis_range:
        # If no new range but stored range exists, use that
        xaxis_range = stored_xaxis_range
        
    else:
        xaxis_range = [occ_time_series["datetime"].min(), occ_time_series["datetime"].max()]
        
    
    ########## OccFeatureEngineer ##########
    occ_time_series = OccFeatureEngineer(
        occ_time_series, 
        course_dates_data, 
        course_info_data, 
        dfguru,
        frequency
    ).derive_features(
        features=global_feature_list, 
        room_id=room_dropdown_value
    )
    
    fig = go.Figure()
    # resize figure
    fig.update_layout(
        height=750,
    )

    inverted_lecture = (~occ_time_series["lecture"].astype(bool)).astype(int)
    y = occ_time_series["occrate"] * inverted_lecture
    
    fig.add_trace(
        go.Scattergl(
            x=occ_time_series["datetime"],
            y=y,
            mode="lines+markers",
            name=f"Occupancy Counts"
        )
    )
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1d",
                        step="day",
                        stepmode="backward"),
                    dict(count=7,
                        label="7d",
                        step="day",
                        stepmode="backward"),
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            range=xaxis_range,
            type="date"
        )
    )
    
    return fig, xaxis_range
    
@callback(
    Output("course_dates_table", "data"),
    Input("xaxis_range_store", "data"),
    Input("ts_viewer_room_filter", "value")
)
def update_course_dates_table(xaxis_range, room_dropdown_value):
    
    # filter by room
    filtered_data = course_dates_data[course_dates_data["room_id"] == room_dropdown_value]
    
    if xaxis_range is None:
        # If no range is set, return the entire dataset or empty
        raise Exception("I am here")
    
    else:
        # Filter course_dates_data based on xaxis_range
        min_time, max_time = xaxis_range
        # Ensure min_time and max_time are datetime objects
        min_time = pd.to_datetime(min_time)
        max_time = pd.to_datetime(max_time)
        
        # Assuming 'datetime' is the column in course_dates_data that has the dates
        mask = (course_dates_data['start_time'] >= min_time) & (course_dates_data['end_time'] <= max_time)
        # Optionally filter by room_id if needed
        filtered_data = filtered_data.loc[mask]
        
    filtered_data = filtered_data.sort_values(by="start_time")
    filtered_data = filtered_data.to_dict('records')
    return filtered_data