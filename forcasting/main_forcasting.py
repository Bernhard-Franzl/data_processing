
from forcasting import DataHandler, Plotter

# TODO:
# Continue with plot seasonal patterns

handler = DataHandler(
    path_repo="data/cleaned_data", 
    file_name="frequency_data", 
)
plotter = Plotter(save_path="forcasting/plots/plot")

data = handler.read_data()

data = handler.filter_by_room_id(data, 1)
data = handler.filter_by_event_type(data, 1)
#data_resampled = handler.resample(data, "datetime", "15min", "count")













"""
# Plot Basic Time Series
plotter.plot_time_series(
    data=data_resampled, 
    x_col="datetime", 
    y_col="event_type", 
)
"""
"""
# Plot Time Series Patterns
plotter.plot_time_series_patterns(
    data=data_resampled, 
    save=True
)
"""