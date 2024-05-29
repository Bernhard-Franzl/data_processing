from webcrawler import Snail
import requests
import re
import pandas as pd

snail = Snail()

# Get the course catalog
#course_catalog = snail.get_detailed_course_catalogue()


#dropdown_entries = snail.search_html(course_catalog, "select", {"name":"room"}, all=False).find_all("option")
#room_dict = {x.text.strip():x["value"] for x in dropdown_entries}

#action, payload = snail.prepare_catalogue_search(course_catalog)

#TODO:
# - derive regularity of course from Kusss!!

room = "HS 19"
df_courses, df_dates = snail.get_courses_by_room(room)

snail.export_to_csv(df_courses, f"{room}_courses.csv")
snail.export_to_csv(df_dates, f"{room}_dates.csv")



