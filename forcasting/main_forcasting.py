
from forcasting import DataHandler, Plotter

# TODO:
# -- Look into resampling -> fix function in forecasting notebook -> not quite there yet!

# -- Deploy probabilistic model from paper
# -- Investigate other ways to clean data
# --
# -- Continue with plot seasonal patterns

handler = DataHandler(
    path_repo="data/cleaned_data", 
    file_name="frequency_data", 
)
plotter = Plotter(save_path="forcasting/plots/plot")

data = handler.read_data()

data = handler.filter_by_room_id(data, 1)
data = handler.filter_by_event_type(data, 1)
#data_resampled = handler.resample(data, "datetime", "15min", "count")






