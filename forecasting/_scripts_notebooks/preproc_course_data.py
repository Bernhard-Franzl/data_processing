from _preprocessing import CoursePreprocessor

room_to_id ={"HS18":0, "HS 18":0, "HS19":1, "HS 19": 1}
door_to_id = {"door1":0, "door2":1}

path_to_raw_courses = "/home/berni/github_repos/data_processing/data/raw_data_22_10"
preprocessor = CoursePreprocessor(path_to_raw_courses, 
                                  room_to_id=room_to_id, door_to_id=door_to_id)


cleaned_course_info, cleaned_course_dates = preprocessor.apply_preprocessing()

# Exam: Exam, Retry, Klausur, NK,
# Test: Test
# Tutorium: Tutorium
#mask = cleaned_course_dates["note"].str.contains("Klausur", case=False)


import re
import numpy as np
# Define regular expressions for each category
exam_pattern = re.compile(
    r'\b(exam|final exam|klausur|teilklausur|klausurtermin|endklausur|TK|NK|nachklausur|retry|zwischenklausur|vorlesungsprüfung|prüfung|resit|hauptklausur|schlussklausur|prüfungs?termin|gesamtklausur|absch(lu|u)ss?klausur|retake)\b',
    re.IGNORECASE
)

tutorium_pattern = re.compile(
    r'\b(tutorium|repetitorium|fragestunde|sprechstunde|übungen|feedback|einsicht|q & a|q&a|q and a|discussion)\b',
    re.IGNORECASE
)

test_pattern = re.compile(
    r'\b(test|teiltest|übungstest|gesamttest|nachtest|moodle-test)\b',
    re.IGNORECASE
)

# Define a regular expression for canceled dates
canceled_pattern = re.compile(
    r'\b(abgesagt|entfällt|fällt aus|ausgefallen|cancel(?:ed)?|termin entfällt|verschoben|postponed)\b',
    re.IGNORECASE
)


# Define a regular expression for offsite courses
offsite_pattern = re.compile(
    r'\b(online|zoom|moodle|video|self[- ]?study|selbststudium|distance learning|remote|aufzeichnung|homework|home work|e-learning|virtual|digital|hybrid|flipped classroom)\b',
    re.IGNORECASE
)

# Function to classify course notes
def classify_course_notes(notes):
    results = []
    for note in notes:
        # Check for matches in each category
        is_exam = bool(exam_pattern.search(note))
        is_tutorium = bool(tutorium_pattern.search(note))
        is_test = bool(test_pattern.search(note))
        is_canceled = bool(canceled_pattern.search(note))
        is_offsite = bool(offsite_pattern.search(note))
        
        # Append the classification to results
        results.append((is_exam, is_tutorium, is_test, is_canceled, is_offsite))
    return results

classified_notes = np.array(classify_course_notes(cleaned_course_dates["note"].values))

cleaned_course_dates["exam"] = classified_notes[:, 0]
cleaned_course_dates["tutorium"] = classified_notes[:, 1]
cleaned_course_dates["test"] = classified_notes[:, 2]
cleaned_course_dates["cancelled"] = classified_notes[:, 3]

# if room == zoom or online 
mask_offsite = np.array(classified_notes[:, 4] | cleaned_course_dates["room"].str.contains("zoom|online", case=False))
cleaned_course_dates["offsite"] = mask_offsite

preprocessor.save_to_csv(cleaned_course_info, "data/lecture_forecasting", "course_info")
preprocessor.save_to_csv(cleaned_course_dates, "data/lecture_forecasting", "course_dates")
