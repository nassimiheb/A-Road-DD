import os 
from datetime import datetime
import csv
import pandas as pd
from flask import send_file, Response
import urllib.request

def trip_name_to_numbers(file):
    file = file[:-4]
    file_split = file.split('_')
    return {
        'id' : file,
        'name': file_split[0],
        "date" : datetime.strptime(file_split[1], '%Y%m%d%H%M%S')
    } 

def get_file_name(name, date):
    return  name + '_' +date.strftime('%Y%m%d%H%M%S') + '.csv'

def read_all_trips(TRIPS_FOLDER):
    results = {}
    for file in os.listdir(TRIPS_FOLDER):
        if file.endswith(".csv"):
            res = trip_name_to_numbers(file)
            results[res['id']] = res
    return results

def create_trip(TRIPS_FOLDER, name):
    date_ = datetime.now()
    file_name = get_file_name(name, date_)
    with open(TRIPS_FOLDER + '/' + file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(('class', 'cor_1', 'cor_2', 'cor_3', 'cor_4', 'lat', 'long'))

    return {
        'id' : file_name[:-4],
        'name': name,
        "date" : date_
    }

def delete_trip(TRIPS_FOLDER, id):
    try:
        os.remove(TRIPS_FOLDER + '/' + id + '.csv')
        return True
    except OSError:
        return False

def get_stats(TRIPS_FOLDER, id):
    try:
        df = pd.read_csv(TRIPS_FOLDER + '/' + id + '.csv')
        list_of_detections = list(df.T.to_dict().values())
        results = {}
        max = 0
        for det in list_of_detections:
            if det['class'] in results:
                results[int(det['class'])] += 1
            else:
                results[int(det['class'])] = 1
            if det['class'] > max: max = det['class'] 
        
        result_list = [0 for i in range(int(max+1))]

        for key, value in results.items():
            result_list[key] = value

        return {
            'data' : list_of_detections,
            'counts': result_list
        }
    except FileNotFoundError: 
        return False
   
def get_image(IMAGES_FOLDER, id, detection_number, app):
    filename = IMAGES_FOLDER + '/' + id + '/' + detection_number + '.jpg'
    try:
        app.send_static_file(filename)
        return 'static/' + IMAGES_FOLDER + '/' + id + '/' + detection_number + '.jpg'
    except FileNotFoundError:
        return False
      
    