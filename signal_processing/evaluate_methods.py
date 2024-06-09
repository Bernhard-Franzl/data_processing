from preprocessing import Preprocessor
from utils import ParameterSearch, Evaluator
#########  Constants #########
room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}
data_path = "/home/berni/data_06_06"


####### Parameter Search ########
path_to_json = "signal_processing/parameters.json"
comb_iterator = ParameterSearch(path_to_json=path_to_json).combinations_iterator(tqdm_bar=True)


def write_results_to_file(comb_number, params, mse, ae):
    with open("results.txt", "a") as file:
        file.write(f"######## Combination: {comb_number} ########\n")
        file.write(f"Parameters: {params}\n")
        file.write(f"MSE: {mse}\n")
        file.write(f"AE: {ae}\n")
        file.write("\n")
        file.write("\n")
        
    return None

for i, params in enumerate(comb_iterator):

    if i == 0:
        old_params = params
        
    # check if for cases where the combination can be skipped
    
    cleaned_data = Preprocessor(data_path, room_to_id, door_to_id).apply_preprocessing(params)
    
    mse, mae = Evaluator("SignalAnalyzer", "data/zählung.csv").evaluate_signal_analyzer(data=cleaned_data, params=params)
    
    write_results_to_file(i, params, mse, mae)
    
# implement a measure to prune unnecessary combinations
# E.g.: if handle 5 and handle 6 are false --> all the combinations in the handle_56_params dict are not necessary
# E.g.: if filter_mode == "discard" --> only 2 parameters are necessary
# We need to make it as modular as possible!

########  Data Preprocessing #########       


# cleaned_data = Preprocessor(data_path, room_to_id, door_to_id).apply_preprocessing(preprocessing_params)
# cleaned_data.to_csv("data/cleaned_data.csv")

# #cleaned_data = pd.read_csv("data/cleaned_data.csv", header=0, index_col=0)
# #cleaned_data["time"] = pd.to_datetime(cleaned_data["time"])


# #########  Data Analysis #########
# # evaluate methods of the SignalAnalyzer class
# # manual control data

# manual_control = pd.read_csv("data/zählung.csv")
# manual_control["room_id"] = manual_control["room"].map(room_to_id)
# mse = 0
# ae = 0
# for index, row in manual_control.iterrows():
#     room_id = row["room_id"]

#     start_time_int = row["start_time"]
#     end_time_int = row["end_time"]
    
#     time_string = row["time"].split(":")
    
#     date = row["date"].split(".")
#     day = int(date[0])
#     month = int(date[1])
    
#     control_people_in = row["people_in"]
    
#     control_time = dt(2024, month, day, int(time_string[0]), int(time_string[1]), 0)
#     start_time = dt(2024, month, day, start_time_int//100, start_time_int%100, 0)
#     end_time = dt(2024, month, day, end_time_int//100, end_time_int%100, 0)
    
#     first = bool(row["first"])
#     last = bool(row["last"])
    
#     analyzer = SignalAnalyzer()
#     data_analysis = analyzer.filter_by_room(cleaned_data, room_id)
    
#     delta = timedelta(minutes=30)
#     save_df = analyzer.filter_by_time(data_analysis, start_time-delta, end_time+delta)
    
    
#     # m is an extremely important parameter -> the one that is used to calculate the extrema
#     df_list, participants, extrema, df_list_plotting, control = analyzer.calc_participants(data_analysis, 
#                                             start_time=start_time,
#                                             end_time=end_time,
#                                             first=first,
#                                             last=last,
#                                             control=True,
#                                             mode="median")
    
    
#     df_plotting = analyzer.merge_participant_dfs(df_list_plotting)
    
#     horizontal_lines = [(participants[0], "black", " before"),
#                         (participants[1], "gold", " after")]

#     vertical_lines = [(control_time, "green", "start"),]
    
#     title = f"Control:{control_people_in}, Time:{control_time}"    
#     analyzer.plot_participants_algo(file_name = f"{control_time}_{room_id}.png",
#                              participants=df_plotting,
#                              df_list = None,
#                              control = None,
#                              extrema = None,
#                              horizontal_lines=[],
#                              vertical_lines=vertical_lines,
#                              title=title)

#     control_row = df_plotting[df_plotting["time"] == control_time]
    
#     #prediction = control_row["people_inside"].values[0]
#     #Mode: Mean 
#     #MSE:  88.55172413793103
#     #AE:  4.0
#     #Mode: Median
#     #MSE: 88.55172413793103
#     #AE: 4.0
    
#     #prediction = int(np.mean(participants))
#     # Mode: Mean
#     #MSE:  32.10344827586207
#     #AE:  3.68965517241379
#     #Mode: Median
#     #MSE: 31.689655172413794
#     #AE:  3.6206896551724137
    
#     prediction = int(np.max(participants))
#     # Mode: Mean
#     #MSE:  32.10344827586207
#     #AE:  3.68965517241379
#     #Mode: Median
#     #MSE:  11.96551724137931
#     #AE:  2.793103448275862
    
#     #prediction = int(np.min(participants))
#     #Mode: Mean, filter
#     #MSE:  100.89655172413794
#     #AE:  5.0344827586206895
#     #Mode: Median
#     #MSE:  105.10344827586206
#     #AE:  5.24137931034482
    
#     # try first of participants
#     #prediction = participants[0]
#     # try second of participants
#     #prediction = participants[1]
    
#     mse_term = (control_people_in - prediction)**2
#     mse += mse_term
#     ae_term = abs(control_people_in - prediction)
#     ae += ae_term
    
    
#     if mse_term > 10:
#         print("##################")
#         print("Time: ", control_time)
#         print("Room: ", room_id)
#         print("Participants: ", participants)
#         print("Control: ", control_people_in)
#         print("Prediction: ", prediction)
#         print("MSE: ", mse_term)
#         print("AE: ", ae_term)
#         print("##################")
#         print()

# print("MSE: ", mse/len(manual_control))
# print("AE: ", ae/len(manual_control))