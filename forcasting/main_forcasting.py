import numpy as np
import pandas as pd
from plots import Plotter
from data import DataHandler, PLCount
import datetime

# TODO:
# -- Implement plots that show the need for data cleaning with plcount
# -- Continue with plot seasonal patterns

handler = DataHandler(
    path_repo="data/cleaned_data", 
    file_name="frequency_data", 
)
data = handler.read_data()

data_filtered = handler.filter_by_room_id(data, 0)
# filter the first two days
data_filtered = handler.filter_by_timestamp(data_filtered, "datetime",
                                            datetime.datetime(2024, 4, 8), 
                                            datetime.datetime(2024, 4, 13))

plcount = PLCount()
frequency = "15min"
occ_list = plcount.run_on_whole_dataset(data_filtered, handler, frequency)

occupancy_counts_raw = handler.calc_occupancy_count(data_filtered, "datetime", frequency)

plotter = Plotter(save_path="plots/image.png",
                  plot_height=600,
                  plot_width=2000)

plotter.plot_plcount(occupancy_counts_raw, pd.concat(occ_list).drop_duplicates(), save=False)
















#data_test = np.array([
#    [0, 1.00, 0],
#    [1, 1.00, 1],
#    [0, 1.00, 1],
#    [1, 1.00, 2],
#    [0, 1.00, 2],
#    [0, 1.00, 2],
#    [2, 1.41, 4],
#    [0, 1.00, 4],
#    [3, 1.73, 7],
#    [1, 1.00, 8],
#    [-2, 1.41, 6],
#    [-7, 2.65, -1],
#    [0, 1.00, -1]
#])
#n=data_test.shape[0]
#m=int(data_test[:,2].max())
#estimates = plcount.run_algorithm(n, m, data_test[:,0], data_test[:,1])
#print(estimates)



