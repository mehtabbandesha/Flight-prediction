import pandas as pd
import numpy as np


def convert_journey(df):
    df['Journey_day'] = pd.to_datetime(df['Date_of_Journey'],format = '%d/%m/%Y').dt.day
    df['Journey_month'] = pd.to_datetime(df['Date_of_Journey'],format = '%d/%m/%Y').dt.month
    df.drop(['Date_of_Journey'],axis=1,inplace=True)
    return df


def convert_departure(df):
    df['Dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
    df['Dep_min'] = pd.to_datetime(df['Dep_Time']).dt.minute
    df.drop(['Dep_Time'],axis=1,inplace=True)
    return df


def convert_arrival(df):
    df['Arrival_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
    df['Arrival_min'] = pd.to_datetime(df['Arrival_Time']).dt.minute
    df.drop(['Arrival_Time'],axis=1,inplace=True)
    return df


def time_taken(df):
    duration = list(df['Duration'])
    
    for i in range(len(duration)):
        if len(duration[i].split()) != 2:
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"
            else:
                duration[i] = "0h " + duration[i]
    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep="h")[0]))
        duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))
    df['Duration_hours'] = duration_hours
    df['DUration_mins'] = duration_mins
    df.drop(['Duration'],axis=1,inplace=True)
    return df


def replace_stops(df):
    df.replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace = True)
    return df

#### handling categorical data

def to_categorical(df):
    Airline = df[['Airline']]
    Airline = pd.get_dummies(Airline,drop_first=True)
    
    ### source column
    Source = df[['Source']]
    Source  = pd.get_dummies(Source,drop_first=True)
    
    ## destination column
    Destination = df[["Destination"]]
    Destination = pd.get_dummies(Destination,drop_first=True)
    
    ## concatenating all dataframe
    df1 = pd.concat([df,Airline,Source,Destination],axis=1)
    return df1





    
