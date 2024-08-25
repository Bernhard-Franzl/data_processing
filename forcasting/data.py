import pandas as pd
import os
import numpy as np
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm

class DataFrameGuru():
    
    def __init__(self, **kwargs):
            
        self.kwargs = kwargs
    
    ######## Read Data ########
    def read_data(self, path_repo, file_name, extension="csv", dtypes=True):
        
        self.path_repo = path_repo
        self.file_name = file_name
        self.dtypes = dtypes
        
        if extension is None:
            self.extension = "csv"
        else:
            self.extension = extension
            
        path = os.path.join(self.path_repo, self.file_name)
        
        data = pd.read_csv(f"{path}.{self.extension}")
        
        if self.dtypes:
            dtype_array = pd.read_csv(f"{path}_dtypes.{self.extension}", index_col=0)
            data = data.astype(dtype_array.to_dict()["0"])
        
        return data
    
    def clean_signal_data(self, dataframe):
        dataframe["datetime"] = dataframe["time"]
        dataframe.drop("time", axis=1, inplace=True)
        return dataframe

    ######## Filter Data ########
    def filter_by_roomid(self, dataframe, room_id):
        return dataframe[dataframe["room_id"] == room_id]
    
    def filter_by_eventtype(self, dataframe, event_type):
        return dataframe[dataframe["event_type"] == event_type]

    def filter_by_timestamp(self, dataframe, time_column, start_time, end_time):
        # only show courses betwen start and end time
        df = dataframe[(dataframe[time_column] >= start_time) & (dataframe[time_column] <= end_time)]
        df = df.sort_values(by=time_column).reset_index(drop=True)
        return df
    
    def filter_by_date(self, dataframe, time_column, date):
        return dataframe[dataframe[time_column].dt.date == date]
    
    def filter_by_daterange(self, dataframe, time_column, start_date, end_date):
        return dataframe[(dataframe[time_column].dt.date >= start_date) & (dataframe[time_column].dt.date <= end_date)]
    
    ######## Resample Data ########
    def _resample(self, dataframe, time_column, frequency, agg_func, output_columns):
        
        # get min day
        if frequency == "MS":
            min_time = min(dataframe[time_column]).replace(day=1, hour=0, minute=0, second=0)
        else:
            min_time = min(dataframe[time_column]).replace(hour=0, minute=0, second=0) 

        # get max day
        max_time = max(dataframe[time_column])
        max_time = max_time.replace(hour=0, minute=0, second=0) + DateOffset(days=1)
        
        
        idx = pd.date_range(start=min_time, end=max_time, freq=frequency, inclusive="both") 

        df_resampled = dataframe.set_index(time_column)\
                    .resample(frequency, label="right", closed="right")

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
        
        df_resampled = df_resampled.rename(columns={"people_inside": "CC"}).drop(["people_in", "people_out"], axis=1)
    
        return df_resampled

    ######## Derive Features ########
    def derive_in_out(self, dataframe):
        
        dataframe["people_in"] = dataframe["event_type"].apply(lambda x: 1 if x == 1 else 0)
        dataframe["people_out"] = dataframe["event_type"].apply(lambda x: 1 if x == 0 else 0)
        
        return dataframe

    def derive_day(self, dataframe, time_column):
        dataframe["day"] = dataframe[time_column].dt.date
        return dataframe

    def derive_week(self, dataframe, time_column):
        dataframe["week"] = dataframe[time_column].dt.isocalendar().week
        return dataframe

    def derive_time(self, dataframe, time_column):
        dataframe["time"] = dataframe[time_column].dt.time
        return dataframe

    def derive_weekday(self, dataframe, time_column):
        dataframe["weekday"] = dataframe[time_column].dt.weekday
        dataframe["weekday_name"] = dataframe[time_column].dt.day_name()
        return dataframe
    
    
class PLCount():
    
    def __init_(self):
        pass
        
    def initialize_algorithm(self, n, m):
        M = np.zeros((n, m+1))
        N = np.zeros((n, m+1))
        M[0,0] = 1
        return M, N
         
    def calc_delta(self, dataframe, column):
        return dataframe[column].diff().fillna(0)
    
    def calc_sigma(self, dataframe, column):
        sigma = dataframe[column].apply(lambda x : np.sqrt(np.abs(x)))
        try:
            sigma = sigma.replace(0, min(sigma[sigma > 0]))
        except:
            sigma = sigma.replace(0, 1)
            
        return sigma
    
    def probability_function(self, c_j, delta_c, sigma):
        exponent = -(c_j - delta_c)**2 / (2 * sigma**2)
        normalizer = 1 / (sigma * np.sqrt(2 * np.pi))
        return normalizer * np.exp(exponent)
    
    def calculate_probability_matrix(self, M, N, delta_array, sigma_array):
        
        for i in range(1, M.shape[0]): # time
            delta_c_i = delta_array[i]
            sigma_i = sigma_array[i]
                
            for j in range(0, M.shape[1]): # count
                
                listy = [self.probability_function(j-k, delta_c_i, sigma_i) * M[i-1, k] for k in range(0, M.shape[1])]
                k_max = np.argmax(listy)
                
                M[i, j] = listy[k_max]
                N[i, j] = k_max

            # normalize row  
            M[i] = M[i] / sum(M[i])
            
        return M, N
    
    def backtracking_zero_init(self, M, N):
        CC_t_n = 0
        occupancy_estimates = np.zeros(M.shape[0])

        for i in range(M.shape[0]-1, 0, -1):
            
            CC_t_n1 = N[i, int(CC_t_n)]
            occupancy_estimates[i-1] = CC_t_n1
            
            CC_t_n = CC_t_n1
        
        return occupancy_estimates
    
    def run_algorithm(self, n, m, delta_array, sigma_array):
        
        M, N = self.initialize_algorithm(n, m)
        
        M, N = self.calculate_probability_matrix(M, N, delta_array, sigma_array)
        
        occupancy_estimates = self.backtracking_zero_init(M, N)
        
        return occupancy_estimates
    
    def run_on_whole_dataset(self, dataframe, data_handler, frequency):
        
        occupancy_count_list = []
        day_list = list(pd.Series(1, dataframe['datetime']).resample("D").sum().index)
        
        for timestamp in tqdm(day_list):
            
            df_filtered = data_handler.filter_by_timestamp(dataframe, "datetime",
                                                  timestamp, timestamp + DateOffset(days=1))
    
            if df_filtered.empty:  
                idx = pd.date_range(start=timestamp, 
                                    end=timestamp + DateOffset(days=1), 
                                    freq=frequency, inclusive="both")
                occupancy_counts = pd.DataFrame(data=0, 
                                                index=idx, 
                                                columns=["CC", "CC_estimates"]).reset_index().rename(columns={"index": "datetime"})
                occupancy_count_list.append(occupancy_counts)    
                
            else:
                occupancy_counts = data_handler.calc_occupancy_count(df_filtered, "datetime", frequency)
                occupancy_counts["delta_CC"] = self.calc_delta(occupancy_counts, "CC")
                occupancy_counts["sigma"] = self.calc_sigma(occupancy_counts, "delta_CC")

                cc_max = occupancy_counts.CC.max()
                m = int(cc_max + (cc_max*0.2))
                n = len(occupancy_counts.datetime)
                
                estimates = self.run_algorithm(n, m, occupancy_counts["delta_CC"], occupancy_counts["sigma"])
                
                occupancy_counts["CC_estimates"] = estimates
                occupancy_counts = occupancy_counts.drop(columns=["delta_CC", "sigma"])
                
                occupancy_count_list.append(occupancy_counts)
                
        return occupancy_count_list