from _preprocessing import CoursePreprocessor

room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}

path_to_raw_courses = "/home/berni/github_repos/data_processing/data/raw_data_22_10"
preprocessor = CoursePreprocessor(path_to_raw_courses, 
                                  room_to_id=room_to_id, door_to_id=door_to_id)

cleaned_course_info, cleaned_course_dates = preprocessor.apply_preprocessing()

preprocessor.save_to_csv(cleaned_course_info, "data/lecture_forecasting", "course_info")
preprocessor.save_to_csv(cleaned_course_dates, "data/lecture_forecasting", "course_dates")
