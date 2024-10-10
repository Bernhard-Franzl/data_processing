from data import DataHandler

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    ######## Helper Functions ########
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
    
    ######## Basic Plotting Functions ########
    def plot_time_series(self, data, x_col, y_col, name, save=False):
        
        fig = go.Figure()
        fig = self.apply_general_settings(fig)
        
        fig.add_trace(
            go.Scatter(
                x=data[x_col], 
                y=data[y_col],
                mode='lines', 
                name=name
                )
            )
        
        if save:
            fig.write_html(f"{self.save_path}.html")
            
        fig.show(config=self.config)
    
    ####### Plot Preprocessing Results ######
    
    def plot_preprocessing(self, raw_data, processed_data, save, show):
    
        fig = make_subplots(rows=1, cols=3, 
                subplot_titles=("Raw Cumulative", "Reset At Midnight", "PLCount Estimates"))

    
    ########### PL Count Plot ###########
    def plot_plcount(self, raw_dataframe, plcount_dataframe, save, show):
        
        # check either save or show must be True
        
        if (not save) & (not show):
            raise ValueError
        
        fig = make_subplots(rows=1, cols=3, 
                    subplot_titles=("Raw Cumulative", "Reset At Midnight", "PLCount Estimates"))
        
        fig = self.apply_general_settings(fig)
        
        fig.add_trace(
            go.Scatter(
                x=raw_dataframe["datetime"],
                y=raw_dataframe["CC"],
                mode='lines',
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plcount_dataframe["datetime"],
                y=plcount_dataframe["CC"],
                mode='lines',
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=plcount_dataframe["datetime"],
                y=plcount_dataframe["CC_estimates"],
                mode='lines',
            ),
            row=1, col=3
        )
        
        if save:
            fig.write_html(f"{self.save_path}.html")
          
        if show:  
            fig.show(config=self.config)
    
    ######## Advanced Plotting Functions ########
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

    ######### Seasonal Patterns #########
    def plot_seasonal_patterns(self, data, frequency, mode):
        
        data_resampled = self.data_handler.resample(data, "datetime", frequency, "count")
        data_resampled = self.derive_day(data_resampled)
        data_resampled = self.derive_time(data_resampled)
        
        
        
        