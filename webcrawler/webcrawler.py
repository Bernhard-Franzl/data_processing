import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm

class Snail():
    """
    Snail that crawls KUSSS and collects data
    """
    base_url = "https://www.kusss.jku.at/kusss/"
    
    def __init__(self):
        
        # get course catalogue
        self.course_catalogue = self.get_detailed_course_catalogue()
        
        # prepare general search
        self.action, self.payload = self.prepare_catalogue_search(self.course_catalogue)
        
        # get all rooms
        dropdown_entries = self.search_html(self.course_catalogue, "select", {"name":"room"}, all=False)\
                                            .find_all("option")
        self.room_dict = {x.text.strip():x["value"] for x in dropdown_entries}                                    
    
    ####### Basic Methods #######
    def crawl(self, url, parameters=None, parse=True):
        """
        Crawls the given URL and extracts the data
        """
        if parameters is None:
            parameters = {}
            
        response = requests.get(url, params=parameters)
        
        if parse:
            response = BeautifulSoup(response.content, "html.parser")
        return response
    
    def search_html(self, soup, tag, attributes=None, all=True):
        """
        Searches the given soup for the given tag and class_name
        """
        if all:
            return soup.find_all(tag, attrs=attributes)
        else:
            return soup.find(tag, attrs=attributes)

    def filter_by_room(self, dates_dataframe, room):
        """
        Filters the dates_dataframe by the given room
        """
        df = dates_dataframe.copy()
        return df[df["Ort"].str.contains(room)].reset_index(drop=True)
    
    def export_to_csv(self, dataframe, filename):
        """
        Exports the dataframe to a csv file
        """
        try:
            dataframe.to_csv(filename, index=False)
            print("Dataframe successfully exported to", filename)
        except:
            print("Error exporting dataframe to", filename)    
    
    ######### Methods for KUSSS #########
    def get_detailed_course_catalogue(self):
        """
        Returns the detailed course catalog
        """
        #coursecatalogue-start.action?advanced=true
        url = self.base_url + "coursecatalogue-start.action" + "?advanced=true"
        soup = self.crawl(url)
        return soup
    
    def prepare_catalogue_search(self, soup):
        """
        Prepares the search for the course catalog
        """
        
        details_form = self.search_html(soup, "form", {"class":"buttonbarinline"}, all=False)
        action = details_form["action"]
        
        payload = {}
        
        input_fields = []
        input_fields += self.search_html(details_form, "input", {"class":"inputfields"}, all=True)
        input_fields += self.search_html(details_form, "input", {"type":"hidden"}, all=True)
        for field in input_fields:
            payload[field["name"]] = field["value"]
            
        select_fields = self.search_html(details_form, "select", all=True)
        for field in select_fields:
            payload[field["name"]] = "all"
            
        return action, payload
        
    def get_search_result_table(self, action, payload):
        """
        Returns the search results
        """
        result_html = self.crawl(self.base_url + action, payload)
        
        tables = self.search_html(result_html, "table", all=True)
        result_table = self.search_html(tables[-1], "tr", all=True)
        
        return result_table

    def extract_search_results(self, result_table):
        """
        Extracts the search results
        """
        format = result_table[0]
        columns = [x.text.strip() for x in self.search_html(format, "th", all=True)]
        
        link_dict = {}
        dataframe = pd.DataFrame(columns=columns)

        for row in result_table[1:]:
            
            cells = self.search_html(row, "td", all=True)
            
            row = []
            for cell in cells:
                
                text = cell.text.strip()
                
                # clean text
                x = [x for x in re.split("\n|\t", text) if x != ""]
                text = "\n".join(x)
                text = x[0]
                
                link = cell.a
                if link is not None:
                    link = link["href"]
                    link_dict[x[0]] = link

                row .append(text)
                
            dataframe.loc[len(dataframe)] = row

        dataframe["max_students"] = None
        dataframe["registered_students"] = None
        
        return dataframe, link_dict

    def clean_string_dates(self, string):
        return re.split("\n|\t|–", string)
    
    def get_lva_details_and_dates(self, lva_url):

        # harvest all the information from the lva overview page
        lva_page = self.crawl(self.base_url + lva_url)

        
        # get some lva details
        info = self.search_html(lva_page, "tr", attributes={"class":"priorityhighlighted"}, all=False).find_all("td")
        lva_number = info[0].get_text(strip=True)
        max_students = int(info[-4].get_text(strip=True))
        registered_students = int(info[-2].get_text(strip=True))

        
        # extract the dates of the lva
        summary = f"Übersicht aller Termine der Lehrveranstaltung {lva_number}"
        dates_table = self.search_html(lva_page, "table", attributes={"summary":summary}, all=True)[1]
        
        
        # dataframe to store information
        dates_dataframe = pd.DataFrame(columns=["LVA-Nummer", "Wochentag", "Datum", "Startzeit", "Endzeit", "Ort", "Anmerkung"])

        #lva_dates = []
        date_list_uncleaned = dates_table.find_all("tr")[1:]
        
        # filter out all dates not in the desired room

        for idx in range(len(date_list_uncleaned)-1):
            
            date_info = [x.strip() for x in self.clean_string_dates(date_list_uncleaned[idx].get_text()) if x!=""]
                
            if idx%2==0:
                if len(date_info) < 5:
                    helper = [" "] * (5 - len(date_info))
                    date_info += helper

                dates_dataframe.loc[len(dates_dataframe)] =   [lva_number, date_info[0], date_info[1], date_info[2], date_info[3], date_info[4], ""]
                #lva_dates.append([date_info[0], date_info[1], date_info[2], date_info[3], date_info[4], ""])

            else:
                if len(date_info) != 0:
                    dates_dataframe.loc[idx//2, "Anmerkung"] = " ".join(date_info)


        return max_students, registered_students, dates_dataframe

    ######### Application #########
    def validate_room(self, room_name):
        try:
            self.payload["room"] = self.room_dict[room_name]
        except KeyError:
            print("Room not found")
            return False
        return True
    
    def accumulate_course_dates(self, dataframe_courses, link_dict, room):
        
        df_courses = dataframe_courses.copy()
        
        dates_list = []
        for i,row in tqdm(df_courses.iterrows()):
            # extract the lva number and action link
            lva_number = row["LVA-Nr."]
            action = link_dict[lva_number]
            
            # get the details and dates of the lva
            max_students, registered_students, df_dates = self.get_lva_details_and_dates(action)
            # filter the dates by the room
            df_dates = self.filter_by_room(df_dates, room)
            
            # store the dates
            dates_list.append(df_dates)
            # store the max and registered students
            df_courses.loc[i, "max_students"] = max_students
            df_courses.loc[i, "registered_students"] = registered_students

        df_dates = pd.concat(dates_list).reset_index(drop=True)
        
        return df_courses, df_dates
    
    def get_courses_by_room(self, room_name):
        
        if self.validate_room(room_name):
            
            # get the search results
            result_table = self.get_search_result_table(self.action, self.payload)
            # extract the search results
            df_courses, link_dict = self.extract_search_results(result_table)
            
            df_courses, df_dates = self.accumulate_course_dates(df_courses, link_dict, room_name)
            
            return df_courses, df_dates 
        
        else:
            return None, None