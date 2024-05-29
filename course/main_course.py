from course_analysis import CourseAnalyzer
from datetime import datetime
import os

#TODO:
# - Add no of dates per course
# - Derive additional infromation -> weekly, calendar weeks, etc. (see notebook)
# - negative values in people coming late?! -> check if it is a bug
# - Deal with irregular courses

room_name = "HS 19"
data_dir = "/home/franzl/github_repos/data_processing/data"
data_dir_course = data_dir

worker = CourseAnalyzer(room_name=None, 
                        data_dir_course=data_dir_course,
                        path_to_signal=os.path.join(data_dir, "cleaned_data.csv"))

# start and end time of the course in datetime format
start_time = worker.df_signal["time"].min()
end_time = worker.df_signal["time"].max()

## must happen before handling combined courses
df = worker.filter_df_by_timestamp(dataframe=worker.df_combined, 
                                   start_time=start_time, 
                                   end_time=end_time)

## must happen before calculating participants
df = worker.handle_combined_courses(df)

df = worker.add_no_dates(df)


course_numbers = list(df[ "course_number"].unique())[:]

## incorporate relative values
# df_result, df_list, _, _ = worker.calc_course_participants(df, mode="max")

# df_result.drop(columns=["max_students",
#                         "max-min","min_idx",
#                         "min_diff_indx","overlength"], inplace=True)

# print(df_result["start_time"])

# worker.export_csv(df_result, "data/df_participants.csv")
# worker.export_metadata("data/metadata_participants.json", 
#                        start_time=start_time, end_time=end_time,
#                        course_numbers=course_numbers)

    