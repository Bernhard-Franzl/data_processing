import pandas as pd
import os

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
    
    ######## Read Data ########
    def read_data(self):
        
        path = os.path.join(self.path_repo, self.file_name)
        
        data = pd.read_csv(f"{path}.{self.extension}")
        
        if self.dtypes:
            dtype_array = pd.read_csv(f"{path}_dtypes.{self.extension}", index_col=0)
            data = data.astype(dtype_array.to_dict()["0"])
            
        data["datetime"] = data["time"]
        data.drop("time", axis=1, inplace=True)
        
        return data

    ######## Filter Data ########
    def filter_by_room_id(self, dataframe, room_id):
        return dataframe[dataframe["room_id"] == room_id]
    
    def filter_by_event_type(self, dataframe, event_type):
        return dataframe[dataframe["event_type"] == event_type]

    ######## Resample Data ########
    def _resample(self, dataframe, time_column, frequency, agg_func, output_columns):
        
        # get min day
        if frequency == "MS":
            min_time = min(dataframe[time_column]).replace(day=1, hour=0, minute=0, second=0)
        else:
            min_time = min(dataframe[time_column]).replace(hour=0, minute=0, second=0)

        # get max day
        max_time = max(dataframe[time_column])
        max_time = max_time.replace(day=max_time.day+1, hour=0, minute=0, second=0)
        
        
        idx = pd.date_range(start=min_time, end=max_time, freq=frequency, inclusive="both") 

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
        
        return df_resampled.reindex(idx, fill_value=0).reset_index().rename(columns={"index": time_column})[[time_column] + output_columns]
        
    def resample(self, dataframe, time_column, frequency, agg_func, output_columns):
        # check if frequency is valid
        if frequency == "1w":
            frequency = pd.tseries.offsets.Week(weekday=0)

        return self._resample(dataframe, time_column, frequency, agg_func, output_columns)
    
    ######## Derive Occupancy Counts ########
    def calc_inside(self, dataframe):

        # calc people entering and leaving
        dataframe["people_inside"] = dataframe["people_in"].cumsum() - dataframe["people_out"].cumsum()
        
        return dataframe

    def calc_occupancy_count(self, dataframe, time_column, frequency):
    
        df = dataframe.copy()
        
        df = self.derive_in_out(df)
        
        df_resampled = self.resample(df, time_column, frequency, "sum", output_columns=["people_in", "people_out"])
        
        df_resampled = self.calc_inside(df_resampled)
    
        return df_resampled

    ######## Derive Features ########
    def derive_in_out(self, dataframe):
        
        dataframe["people_in"] = dataframe["event_type"].apply(lambda x: 1 if x == 1 else 0)
        dataframe["people_out"] = dataframe["event_type"].apply(lambda x: 1 if x == 0 else 0)
        
        return dataframe

    def derive_day(self, dataframe):
        dataframe["day"] = dataframe["datetime"].dt.date
        return dataframe

    def derive_week(self, dataframe):
        dataframe["week"] = dataframe["datetime"].dt.isocalendar().week
        return dataframe

    def derive_time(self, dataframe):
        dataframe["time"] = dataframe["datetime"].dt.time
        return dataframe

    def derive_weekday(self, dataframe):
        dataframe["weekday"] = dataframe["datetime"].dt.weekday
        return dataframe
    