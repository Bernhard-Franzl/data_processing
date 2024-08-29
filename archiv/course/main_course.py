from course_analysis import CourseAnalyzer
from datetime import datetime
import os

#TODO:
# - better way of caluclating the course participants -> somehow look at the dynamics in the data 
# -> method fails if course terminates too long before the end of the course -> then the median is not a good indicator
# -> we  need better method to handle during the course!

# - or maybe try the curve fitting approach, distribution fitting, etc.

#room_name = "HS 19"
#data_dir = "/home/franzl/github_repos/data_processing/data"

data_dir = "/home/berni/github_repos/data_processing/data/cleaned_data"
data_dir_course = data_dir

worker = CourseAnalyzer(path_to_courses=data_dir,
                        path_to_signal=data_dir)

# start and end time of the course in datetime format
start_time = worker.df_frequency_data["time"].min()
end_time = worker.df_frequency_data["time"].max()
print(start_time, end_time)

## must happen before handling combined courses
df = worker.filter_df_by_timestamp(dataframe=worker.df_combined, 
                                   start_time=start_time, 
                                   end_time=end_time)


## must happen before calculating participants
df = worker.handle_combined_courses(df)
# must happen after time filtering
df = worker.add_no_dates(df)

course_numbers = list(df[ "course_number"].unique())[:]
# incorporate relative values
df_result, df_list, extrema_list, df_plot_list = worker.calc_course_participants(df)

df_result.drop(columns=["max_students",
                        "max-min","min_idx",
                        "min_diff_indx"], inplace=True)

print(len(df_result))
worker.export_csv(df_result, "data/webapp_data/df_participants.csv")
worker.export_metadata("data/webapp_data/metadata_participants.json", 
                    start_time=start_time, end_time=end_time,
                    course_numbers=course_numbers)

    