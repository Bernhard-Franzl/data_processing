import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

class DataPlotter():
    
    def __init__(self, save_path, dataframe_guru, *args, **kwargs):
        
        self.df_guru = dataframe_guru
        
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
    
    def plot_some_features(self, data, save=False):
        
        fig = make_subplots(
            rows=3, 
            cols=1, 
            subplot_titles=("Occupancy Count", "Registered Students", "Temperature in Linz"),
            vertical_spacing=0.2,
            )
        
        fig = self.apply_general_settings(fig)
        
        x_col = "datetime"
        
        # occupancy rate
        fig.add_trace(
            go.Scatter(
                x=data[x_col], 
                y=data["occcount"],
                mode='lines', 
                name='Occupancy Count'
                ),
            row=1, col=1
            )
        
        # registered students
        fig.add_trace(
            go.Scatter(
                x=data[x_col], 
                y=data["registered"] ,
                mode='lines', 
                name='Registered Students'
                ),
            row=2, col=1
            )

        # temperature
        fig.add_trace(
            go.Scatter(
                x=data[x_col], 
                y=data["tl"],
                mode='lines', 
                name='Temperature in Linz'
                ),
            row=3, col=1
            )

        # set y axis between 0 and 1
        fig.update_yaxes(range=[-5, 60], row=1, col=1)
        fig.update_yaxes(range=[-15, 220], row=2, col=1)
        fig.update_yaxes(range=[6, 12], row=3, col=1)
        
        # margin settings
        fig.update_layout(margin=dict(l=75, r=50, t=50, b=50))
        
        # label the y-axis 
        fig.update_yaxes(title_text="Occupancy Count", title_font=dict(size=13), row=1, col=1)
        fig.update_yaxes(title_text="Registered Students", title_font=dict(size=13), row=2, col=1)
        fig.update_yaxes(title_text="Temperature [°C]", title_font=dict(size=13), row=3, col=1)
        
        # remove legend
        fig.update_layout(showlegend=False)
            
        fig.show(config=self.config)
        
        return fig
    
    ####### Plot Preprocessing Results ######
    def plot_preprocessing(self, raw_data, processed_data, plcount_data, save_bool, show_bool, combined):
        
        if combined:
            
            self.plot_width = 1000
            fig = make_subplots(
                rows=1, 
                cols=1, 
                subplot_titles=("Raw Data", "Simple Preprocessing Applied", "PLCount Estimates")
                )

            fig.add_trace(
                go.Scatter(
                    x=raw_data["datetime"],
                    y=raw_data["CC"],
                    mode='lines',
                    name="Raw"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=processed_data["datetime"],
                    y=processed_data["CC"],
                    mode='lines',
                    name="Simple Preprocessing"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plcount_data["datetime"],
                    y=plcount_data["CC_estimates"],
                    mode='lines',
                    name="PLCount"
                ),
                row=1, col=1
            )
            
            fig = self.apply_general_settings(fig)
            
            if save_bool:
                fig.write_html(f"{self.save_path}.html")
            
            if show_bool:  
                fig.show(config=self.config)
                              
        else:
            n_col = 1
            n_row = 2
            self.plot_height = 600
            self.plot_width = 800
            

            fig = make_subplots(
                rows=n_row, 
                cols=n_col,
                shared_yaxes=True,
                vertical_spacing=0.2,
                #specs=[[{}, {}],
                #        [{"colspan": 2}, None]],
                subplot_titles=("Raw Occupancy Count",
                                #"Error Correction Applied", 
                                "Preprocessing Pipeline Applied"))
            fig.update_annotations(font_size=20)
            
            fig.add_trace(
                go.Scatter(
                    x=raw_data["datetime"][:-1],
                    y=raw_data["CC"][:-1],
                    mode='lines',
                    name="Raw"
                ),
                row=1, col=1
            )
            
            #fig.add_trace(
            #    go.Scatter(
            #        x=processed_data["datetime"][:-1],
            #        y=processed_data["CC"][:-1],
            #        mode='lines',
            #        name="Simple Preprocessing"
            #    ),
            #    row=1, col=2
            #)
            
            fig.add_trace(
                go.Scatter(
                    x=plcount_data["datetime"],
                    y=plcount_data["CC_estimates"],
                    mode='lines',
                    name="PLCount"
                ),
                row=2, col=1
            )
            
            for x in range(n_row+1):
                fig.update_yaxes(title_text="Occupancy Count",
                                title_font=dict(size=18),
                                showgrid=True, 
                                row=x, col=1)
                        
            for x in range(n_col+1):
                fig.update_xaxes(title_text="Time",
                                 title_font=dict(size=18), 
                                 showgrid=True,
                                 row=1, 
                                 tickformat="%H:%M",
                                 col=x)
                
            fig.update_xaxes(title_text="Time",
                                title_font=dict(size=18), 
                                showgrid=True,
                                row=2, 
                                tickformat="%H:%M",
                                col=1)
                    
            fig = self.apply_general_settings(fig)
            fig.update_layout(showlegend=False)
            
            if save_bool:
                fig.write_image(f"{self.save_path}.png", scale=5)
                fig.write_html(f"{self.save_path}.html")
            
            if show_bool:  
                fig.show(config=self.config)
            
            return fig
             
    ########### PL Count Plot ###########
    def plot_plcount(self, raw_dataframe, plcount_dataframe, save_bool, show_bool):
        
        # check either save or show must be True
        
        if (not save_bool) & (not show_bool):
            raise ValueError("Either save or show must be True")
        
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
        
        if save_bool:
            fig.write_html(f"{self.save_path}.html")
          
        if show_bool:  
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
        
        
        
        