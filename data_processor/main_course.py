from course_analysis import CourseAnalyzer
from datetime import datetime
#from visulization import Visualizer
#from matplotlib import pyplot as plt

#TODO:
# - negative values in people coming late?! -> check if it is a bug
# - column placement not yet correct
# - Add subtitle with additional information
# - Deal with irregular courses
# - Implement proper hover text
# - Fix distance between subplots


room_name = "HS 19"
data_dir_course = "/home/berni/github_repos/webapp/data"
data_dir_signal = "/home/berni/data_05_13"

worker = CourseAnalyzer(room_name=None, 
                        data_dir_course=data_dir_course,
                        data_dir_signal=data_dir_signal)

# start and end time of the course in datetime format
start_time = worker.df_signal["time"].min().replace(hour=0, minute=0, second=0)
end_time = worker.df_signal["time"].max().replace(hour=23, minute=59, second=0)

## must happen before handling combined courses
df = worker.filter_df_by_timestamp(dataframe=worker.df_combined, 
                                   start_time=start_time, 
                                   end_time=end_time)

## must happen before calculating participants
df = worker.handle_combined_courses(df)

course_numbers = list(df[ "course_number"].unique())[:]
## incorporate relative values
df_result, df_list, _, _ = worker.calc_course_participants(df, mode="max")
df_result.drop(columns=["semester", "max_students",
                        "max-min","min_idx",
                        "min_diff_indx","overlength"], inplace=True)

print(df_result["start_time"])
worker.export_csv(df_result, "data/df_participants.csv")
worker.export_metadata("data/metadata_participants.json", 
                       start_time=start_time, end_time=end_time,
                       course_numbers=course_numbers)

# must be given the signal during course time -> wise to do after calc_course_participants
# or we need calc inside per min first

#df_results, entering_students_list, leaving_students_list, attendance_dynamics_list = worker.calc_dynamics_all_dates(df_result, df_list)

#visard = Visualizer()

"""df_plot = df_results.copy()
df_plot["start_time"] = df_plot["start_time"].dt.strftime("%d.%m.%Y %H:%M")
df_plot = worker.filter_df_by_room(df_plot, 0).sort_values(by="late_students", ascending=False)
#df_plot = df_plot.head(20)

print(df_plot.columns[[0, 8, 19, 16, 7, 1, 5, 2, 27, 3]])
df_plot["x"] = df_plot["start_time"]


# relative present
df_plot = visard.calc_relative_present(df_plot, "late_students").sort_values(by="relative_present", ascending=False)
df_plot["y"] = df_plot["relative_present"]
"""

# day
#df["day"] = df["start_time"].dt.strftime("%d.%m.%Y")
#df.groupby(by=["day"])[["late_students", "registered_students", "present_students"]].sum()

#df_plot = visard.late_students_prepare(df_results, 0, mode="grouped", grouped_by="start_time")
#print(df_plot)

#visard.plot_late_students(df_plot)














#df_result = worker.calc_#visard = Visualizer()

#visard.plot_multiple_courses_bars(dataframe=df_result,
#                                  course_numbers=course_numbers,
#                                  title="Participants per Course Date",
#                                  relative=False)


##print(df_result.columns[[0, 8, 19, 16, 7, 1, 5, 2, 27, 3]])
#mode = "relative_registered"
#visard.plot_multiple_courses_bars(dataframe=df_result,
#                                  course_numbers=course_numbers,
#                                  title="Participants per Course Date",
#                                  mode=mode)

# single course bar chart
#visard.plot_course_bar(dataframe=df_result, 
#                       course_number=course_numbers[0], 
#                       show_relative=True, 
#                       show_before_after=False)





#visard.plot_multiple_courses_bars(dataframe=df_result,
#                                  course_numbers=course_numbers,
#                                  title="Participants per Course Date",
#                                  relative=False)


##print(df_result.columns[[0, 8, 19, 16, 7, 1, 5, 2, 27, 3]])
#mode = "relative_registered"
#visard.plot_multiple_courses_bars(dataframe=df_result,
#                                  course_numbers=course_numbers,
#                                  title="Participants per Course Date",
#                                  mode=mode)

# single course bar chart
#visard.plot_course_bar(dataframe=df_result, 
#                       course_number=course_numbers[0], 
#                       show_relative=True, 
#                       show_before_after=False)
































#for idx in range(len(df_result)):
    
#    df_plot_list = plot_list[idx]
#    df_plotting = visard.merge_participant_dfs(df_plot_list)

#    participants = part_list[idx]
#    horizontal_lines = [(participants[0], "black", " before"),
#                        (participants[1], "gold", " after")]


#    dataframes = df_list[idx]
#    extrema = extrema_list[idx]
#    plot_name = plot_name_list[idx]
    
#    title = "Irregular: " + str(df.iloc[idx]["irregular"])
#    visard.plot_participants(save_path = f"plots/{plot_name}.png",
#                             participants=df_plotting,
#                             df_list = dataframes,
#                             control = None,
#                             extrema = None,
#                             horizontal_lines=horizontal_lines,
#                             title=title)

    
    