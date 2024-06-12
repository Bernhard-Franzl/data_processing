from preprocessing import SignalPreprocessor, CoursePreprocessor


#########  Signal Preprocessing #########
room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}
path_to_raw_signal = "/home/berni/data_06_06/archive"

#TODO:
# Function that stores data and datatypes 

#params = {
#    "filtering_params":{
#    "discard_samples": True,
#    "apply_filter": True,
#    "filter_mode": "time_window",
#    "k":5,
#    "nm":3,
#    "ns":1,
#    "s":2,
#    "lb_in":4,
#    "lb_out":3,
#    "handle_5":False,
#    "handle_6":False
#    }
#}
#preprocessor = SignalPreprocessor(path_to_raw_signal, room_to_id, door_to_id)
#cleaned_signal_data, raw_signal_data = preprocessor.apply_preprocessing(params)
#print(cleaned_signal_data)


#path_to_raw_courses = "/home/berni/github_repos/data_processing/data"
#preprocessor = CoursePreprocessor(path_to_raw_courses, 
#                                  room_to_id=room_to_id, door_to_id=door_to_id)
#cleaned_course_info, cleaned_course_dates = preprocessor.apply_preprocessing()

#preprocessor.save_to_csv(cleaned_course_info, path_to_raw_courses, "cleaned_course_info")
#preprocessor.save_to_csv(cleaned_course_dates, path_to_raw_courses, "cleaned_course_dates")
