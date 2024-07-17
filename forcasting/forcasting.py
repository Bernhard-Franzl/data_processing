import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Forcaster():
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def forecast(self, steps):
        return self.model.predict(steps)
    

class DataHandler():
    
    def __init__(self, path_repo, file_name, dtypes=True, extension="csv", **kwargs):
        
        self.path_repo = path_repo
        self.file_name = file_name
        self.dtypes = dtypes
        if extension is None:
            self.extension = "csv"
        else:
            self.extension = extension
            
        self.kwargs = kwargs
    
    def read_data(self):
        
        path = os.path.join(self.path_repo, self.file_name)
        
        data = pd.read_csv(f"{path}.{self.extension}")
        
        if self.dtypes:
            dtype_array = pd.read_csv(f"{path}_dtypes.{self.extension}", index_col=0)
            data = data.astype(dtype_array.to_dict()["0"])
            
        data["datetime"] = data["time"]
        data.drop("time", axis=1, inplace=True)
        
        return data

    def filter_by_room_id(self, dataframe, room_id):
        return dataframe[dataframe["room_id"] == room_id]
    
    def filter_by_event_type(self, dataframe, event_type):
        return dataframe[dataframe["event_type"] == event_type]

    def _resample(self, dataframe, time_column, frequency, agg_func):
        
        # get min day
        if frequency == "MS":
            min_time = min(dataframe[time_column]).replace(day=1, hour=0, minute=0, second=0)
        else:
            min_time = min(dataframe[time_column]).replace(hour=0, minute=0, second=0)

        # get max day
        max_time = max(dataframe[time_column])
        max_time = max_time.replace(day=max_time.day+1, hour=0, minute=0, second=0)
        
        
        idx = pd.date_range(start=min_time, end=max_time, freq=frequency) 

        df_resampled = dataframe.set_index(time_column)\
                    .resample(frequency, label="left", closed="left")

        if agg_func == "sum":
            df_resampled = df_resampled.sum()
        elif agg_func == "count":
            df_resampled = df_resampled.count()
        elif agg_func == "value_counts":
            df_resampled = df_resampled.value_counts()
        else:
            raise ValueError("agg_func must be 'sum', 'count' or 'value_counts'")
        return df_resampled.reindex(idx, fill_value=0).reset_index().rename(columns={"index": time_column})[[time_column, "event_type"]]
    
    def resample(self, dataframe, time_column, frequency, agg_func):
        # check if frequency is valid
        if frequency == "1w":
            frequency = pd.tseries.offsets.Week(weekday=0)

        return self._resample(dataframe, time_column, frequency, agg_func)
    
    def derive_day(self, data):
        data["day"] = data["datetime"].dt.date
        return data

    def derive_week(self, data):
        data["week"] = data["datetime"].dt.isocalendar().week
        return data

    def derive_time(self, data):
        data["time"] = data["datetime"].dt.time
        return data

class Plotter():
    
    def __init__(self, save_path, *args, **kwargs):
        
        self.data_handler = DataHandler(path_repo="", file_name="", *args, **kwargs)
        
        self.save_path = save_path
        
        self.figsize = (10,5)
        
        # font settings
        self.font_family = "Arial, sans-serif"
        
        self.margin=dict(l=50, r=50, t=50, b=50)
        if "margin" in kwargs:
            self.margin = kwargs["margin"]
            
        self.axis_title_size = 18
        if "axis_title_size" in kwargs:
            self.axis_title_size = kwargs["axis_title_size"]
            
        self.text_size = 14
        if "text_size" in kwargs:
            self.text_size = kwargs["text_size"]
            
        self.title_size = 30
        if "title_size" in kwargs:
            self.title_size = kwargs["title_size"]
         
        # format settings
        self.plot_height_provided = 500
        if "plot_height" in kwargs:
            self.plot_height_provided = kwargs["plot_height"]

        self.plot_height = self.plot_height_provided
        
        self.plot_width_provided = 1000
        if "plot_width" in kwargs:
            self.plot_width_provided = kwargs["plot_width"]
            
        self.plot_width = self.plot_width_provided 
        
        # plotly config
        self.config={
            "responsive": True,
            'scrollZoom': True,
            "displaylogo": False,
            "displayModeBar": True,
            #"modeBarButtonsToRemove": 
            #    ["select", "zoomIn", "zoomOut", "autoScale", "lasso2d"]
            }
        
    def apply_general_settings(self, fig):
        
        # general layout settings
        fig.update_layout(
            #barmode='group',
            dragmode='pan',
            font=dict(
                family=self.font_family,
                size=self.text_size,),
            margin=self.margin,
            height=self.plot_height,
            width=self.plot_width,
            hoverlabel=dict(font_size=self.text_size+2))

        #if self.plot_width is not None:
        #    fig.update_layout(width=self.plot_width)
            
        return fig
    
    def plot_time_series(self, data, x_col, y_col, save=False):
        
        fig = go.Figure()
        fig = self.apply_general_settings(fig)
        
        fig.add_trace(
            go.Scatter(
                x=data[x_col], 
                y=data[y_col],
                mode='lines', 
                name='lines'
                )
            )
        
        if save:
            fig.write_html(f"{self.save_path}.html")
            
        fig.show(config=self.config)
      
    def plot_time_series_patterns(self, data, save=False):
        
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=("per Hour", "Per Day", "Per Week", "Per Month"))
        
        
        fig = self.apply_general_settings(fig)
        
        list_of_frequencies = [("1h", "hour"), ("1d", "day"), ("1w", "week"), ("MS", "month")]
        for i, tuple_i in enumerate(list_of_frequencies):
            
            frequency, name = tuple_i
            
            data_resampled = self.data_handler.resample(data, "datetime", frequency, "count")
            
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=data_resampled["datetime"],
                    y=data_resampled["event_type"],
                    mode='lines',
                    name=name
                ),
                row=row, col=col
            )

        if save:
            fig.write_html(f"{self.save_path}.html")
            
        fig.show(config=self.config)
        
    def plot_seasonal_patterns(self, data, frequency, mode):
        
        data_resampled = self.data_handler.resample(data, "datetime", frequency, "count")
        data_resampled = self.derive_day(data_resampled)
        data_resampled = self.derive_time(data_resampled)
        
        
        
        