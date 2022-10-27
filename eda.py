import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates

df_underground = pd.read_csv('data/underground.csv')
df_weather = pd.read_csv('data/weather.csv')

def weatherprep(df):
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  # Weather data is recorded in different time zones, all data needs to be added for 8 hours to change to the correct time zone
  df['DateTime'] = df['DateTime'] + timedelta(hours=8)
  df["month"] = df['DateTime'].dt.month
  df["day"] = df['DateTime'].dt.day
  df["hours"] = df['DateTime'].dt.hour
  df['minutes'] = df['DateTime'].dt.minute
  df["date"] = df['DateTime'].dt.date  
  df = df.groupby(['month','day','hours','minutes']).first().reset_index()
  df = df[['DateTime','WeatherRain']]
  df = df[df['DateTime'].dt.year > 2021] 
  df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:00')
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  df.drop_duplicates(keep='first', inplace=True) 
  df = df.sort_values(by='DateTime')
  df.reset_index(inplace=True)
  df=df.drop(['index'], axis=1)
  return df

df_weather = weatherprep(df_weather)

#function to filter date
def filterdate(df,start,end):
  df = df.set_index('DateTime')
  df = (df[start:end])
  df.reset_index(inplace=True)
  return df

df_weather_sepoct = filterdate(df_weather,start='2022-09-01 00:00',end='2022-10-30 23:00')

