from preprocessing import SignalPreprocessor, CoursePreprocessor
import json

room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}
data_path = "/home/berni/data_29_06_merged/archive"


# test single parameter set
path_to_json = "signal_processing/processing_parameters.json"
params = json.load(open(path_to_json, "r"))

preprocessor = SignalPreprocessor(data_path, room_to_id, door_to_id)
cleaned_data, raw_data = preprocessor.apply_preprocessing(params)

# save data
preprocessor.save_to_csv(cleaned_data, "data/cleaned_data", "frequency_data")


path_to_raw_courses = "/home/berni/github_repos/data_processing/data/raw_data"
preprocessor = CoursePreprocessor(path_to_raw_courses, 
                                  room_to_id=room_to_id, door_to_id=door_to_id)

cleaned_course_info, cleaned_course_dates = preprocessor.apply_preprocessing()

print(cleaned_course_dates[cleaned_course_dates["course_number"] == "256.300"])

preprocessor.save_to_csv(cleaned_course_info, "data/cleaned_data", "course_info")
preprocessor.save_to_csv(cleaned_course_dates, "data/cleaned_data", "course_dates")
