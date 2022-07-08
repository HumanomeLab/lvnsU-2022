import os
import numpy as np
import datetime
import json
from ast import literal_eval
import pandas as pd

def get_Data_activity(dataPath, date, data_collection):

    tmp_datetime = datetime.datetime.strptime(date, '%Y-%m-%d')

    date_list = [tmp_datetime + datetime.timedelta(minutes=tmp) for tmp in range(1440)]
    value_list = [0] * 1440
    if 'time' not in data_collection:
        data_collection['time'] = np.array([])
        data_collection['value'] = np.array([])
    data_collection['time'] = np.append(data_collection['time'], np.array(date_list))
    data_collection['value'] = np.append(data_collection['value'], np.array(value_list))

    with open(dataPath) as f:
        s = f.read()

    if s != '':
        tmp_s = s.replace("'", '"')
        tmpData = json.loads(tmp_s)

        kindOfData = 'activities-heart-intraday'  
        if kindOfData in tmpData:
            for oneMinute_data in tmpData[kindOfData]['dataset']:
                tmp = datetime.datetime.strptime(date + ' ' + oneMinute_data['time'], '%Y-%m-%d %H:%M:%S')
                data_collection['value'][data_collection['time'] == tmp] = oneMinute_data['value']

    return data_collection

def get_Data_sleep(dataPath, data_collection):

    with open(dataPath) as f:
        s = f.read()

    tmp_s = s.replace("'", '"')
    tmp_s = tmp_s.replace('True', '"True"')
    tmp_s = tmp_s.replace('False', '"False"')
    tmpData = json.loads(tmp_s)

    if tmpData['sleep'] != []:
        for ii in range(len(tmpData['sleep'])):
            for sleep_data in tmpData['sleep'][ii]['levels']['data']:
                duration_minutes = sleep_data['seconds'] // 60
                start_datetime = datetime.datetime.strptime(sleep_data['dateTime'].split('.000')[0], '%Y-%m-%dT%H:%M:%S')

                for minute in range(duration_minutes):
                    tmp_datetime = start_datetime + datetime.timedelta(minutes=minute)
                    hour_min = datetime.datetime(tmp_datetime.year, tmp_datetime.month, tmp_datetime.day, tmp_datetime.hour, tmp_datetime.minute)

                    data_collection['sleep'][data_collection['time'] == hour_min] = int(1)

    return data_collection


def get_sleep_hr(data_collection, sleep=True, all=False):
    time = data_collection['time']
    hr_value = data_collection['value']
    sleep_flg = data_collection['sleep']
    
    df = pd.DataFrame(np.array([time, hr_value, sleep_flg]).T, columns=["time", "hr_value", "sleep_flg"])
    
    if not all:
        if sleep:
            df = df[df["sleep_flg"]==1]
        else:
            df = df[df["sleep_flg"]==0]
    
    return df

def get_sleep_hr_data(hr_data_paths, sleep_data_paths):

    data_collection = dict()
    for hr_data_path in hr_data_paths:
        date = os.path.basename(hr_data_path).split('.txt')[0]
        data_collection = get_Data_activity(hr_data_path, date, data_collection)
    
    start_datetime = datetime.datetime.strptime(os.path.basename(sleep_data_paths[0]).split('.txt')[0], '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(os.path.basename(sleep_data_paths[-1]).split('.txt')[0], '%Y-%m-%d') + datetime.timedelta(days=1)
    data_collection['sleep'] = np.array([0] * 1440 * (end_datetime - start_datetime).days)

    sleep_start_end_datetime = dict()
    for sleep_data_path in sleep_data_paths:
        data_collection = get_Data_sleep(sleep_data_path, data_collection)
        date = datetime.datetime.strptime(os.path.basename(sleep_data_path).split('.txt')[0], '%Y-%m-%d')
        with open(sleep_data_path) as file:
            data_text = file.read()
        tmp_s = data_text.replace("'", '"')
        tmp_s = tmp_s.replace('true', '"True"')
        tmp_s = tmp_s.replace('false', '"False"')
        data_dict = json.loads(tmp_s)
        start_time = data_dict['sleep'][0]['startTime'].split('.')[0]
        end_time = data_dict['sleep'][0]['endTime'].split('.')[0]
        start_datetime = datetime.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
        end_datetime = datetime.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')

        sleep_start_end_datetime[date] = {'start':start_datetime, 'end':end_datetime}

    df = pd.DataFrame()
    df['time'] = data_collection['time']
    df['hr_value'] = data_collection['value']
    df['sleep_flg'] = data_collection['sleep']

    return df, sleep_start_end_datetime

def get_RHR(hr_data_paths):
    date_list = []
    rhr_list = []

    for hr_data_path in hr_data_paths:
        #Get date
        date = datetime.datetime.strptime(os.path.basename(hr_data_path).split('.txt')[0], '%Y-%m-%d')
        date_list.append(date)

        #Get RHR value
        with open(hr_data_path) as file:
            hr_data_text = file.read()
        hr_data_dict = literal_eval(hr_data_text)
        rhr_value = hr_data_dict['activities-heart'][0]['value']['restingHeartRate']
        rhr_list.append(rhr_value)

    #Create RHR data frame and output the data frame as csv
    df_rhr = pd.DataFrame()
    df_rhr['Date'] = date_list
    df_rhr['Date'] = pd.to_datetime(df_rhr['Date'])
    df_rhr['RHR'] = rhr_list

    return df_rhr

#=========================
# Get minimum heart rate during sleep per day
#=========================
def get_hr_min_per_day(sleep_data_paths, df_all_sleep_data):
    #Get all sleep data
    sleep_data = dict()
    for sleep_data_path in sleep_data_paths:
        date = os.path.basename(sleep_data_path).split('.txt')[0]
        with open(sleep_data_path) as file:
            data_text = file.read() 
        data_dict = literal_eval(data_text)
        sleep_data[date] = data_dict

    #Get start and end times of sleep per day
    start_and_end_daitetime = dict()

    for date, data in sleep_data.items():
        try:
            start_date = data['sleep'][0]['startTime'].split('T')[0]
            end_date = data['sleep'][0]['endTime'].split('T')[0]
            start_time = data['sleep'][0]['startTime'].split('T')[1].split('.')[0]
            end_time = data['sleep'][0]['endTime'].split('T')[1].split('.')[0]
            start_datetime = datetime.datetime.strptime(start_date + ' ' + start_time, '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.datetime.strptime(end_date + ' ' + end_time, '%Y-%m-%d %H:%M:%S')

            datetime_dict = dict()
            datetime_dict['start'] = start_datetime
            datetime_dict['end'] = end_datetime

            start_and_end_daitetime[date] = datetime_dict
        except IndexError as e:
            print(f'Sleep data of {date} is missing.')
    
    #Get minimum heart rate during sleep per day
    min_dict = dict()
    
    for date, time in start_and_end_daitetime.items():
        start = time['start']
        end = time['end']
        sleep_hr_value = np.array(df_all_sleep_data.loc[start:end]['hr_value'])
        sleep_hr_value_min = sleep_hr_value.min()
        min_dict[date] = sleep_hr_value_min
    
    return min_dict