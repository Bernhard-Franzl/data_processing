from webcrawler import Snail
import requests
import re
import pandas as pd

"""
Note: 
Webcrawler worked on 15.6.2024 for HS 19 and HS 18. 
Due to changes on the website, the webcrawler might not work anymore.
"""
for room in ["HS 18", "HS 19"]:
    snail = Snail()
    df_courses, df_dates = snail.get_courses_by_room(room)

    
    print(df_dates)
    
    raise
    snail.export_to_csv(df_courses, f"data/raw_data_22_10/{room}_courses.csv")
    snail.export_to_csv(df_dates, f"data/raw_data_22_10/{room}_dates.csv")
    
    snail.driver.quit()



