import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
#import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_error, log_loss,r2_score,mean_absolute_error


df_underground = pd.read_csv('data/underground.csv')
df_weather = pd.read_csv('data/weather.csv')

def prep(df):
  df['DateTime']=pd.to_datetime(df['DateTime'])
  df['DateTime'] = df['DateTime'] + timedelta(hours=8)
  df = df[df['DateTime'].dt.year > 2021]
  df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:00')
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  df= df.sort_values(by='DateTime')
  return df 

def weatherprep(df):
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  # Weather data is recorded in different time zones, all data needs to be added for 8 hours to change to the correct time zone
  df['DateTime'] = df['DateTime'] + timedelta(hours=8)
  df["month"] = df['DateTime'].dt.month
  df["day"] = df['DateTime'].dt.day
  df["hours"] = df['DateTime'].dt.hour
  df['minutes'] = df['DateTime'].dt.minute
  df["date"] = df['DateTime'].dt.date
  #df['DateTime'] = df['DateTime'].dt.round('10min')  
  #df = df.groupby(['month','day','hours','minutes']).first().reset_index()
  df = df.groupby(['month','day','hours']).first().reset_index()
  df = df[['DateTime','WeatherRain','WeatherTemp','WeatherHumid','WeatherWindD','WeatherWindS','WeatherSolar']] 
  df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:00:00')
  #df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:00')
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  df.drop_duplicates(keep='first', inplace=True)
  #df.drop_duplicates(subset='DateTime', keep='first') 
  df = df.sort_values(by='DateTime')
  df.reset_index(inplace=True)
  df=df.drop(['index'], axis=1)
  return df

#function to filter date
def filterdate(df,start,end):
  df = df.set_index('DateTime')
  df = (df[start:end])
  df.reset_index(inplace=True)
  return df

def adddate(df):
  df["day"] = df['DateTime'].dt.day
  df["month"] = df['DateTime'].dt.month
  return df

dfEUI = df_underground[df_underground['EUI'].notna()]
dfID = df_underground[df_underground['DeviceID'].notna()]

conditions = [
    (dfID['DeviceID'] == "UDW-001"),
    (dfID['DeviceID'] == "UDW-002"),
    (dfID['DeviceID'] == "UDW-003"),
    (dfID['DeviceID'] == "UDW-004"),
    (dfID['DeviceID'] == "UDW-005"),
    (dfID['DeviceID'] == "UDW-006"),
    (dfID['DeviceID'] == "UDW-007"),
    (dfID['DeviceID'] == "UDW-009"),
    (dfID['DeviceID'] == "UDW-010"),
    (dfID['DeviceID'] == "UDW-011")
]

values = ["24E124126C326140","24E124126C326742","24E124126C326591","24E124126C326567","24E124126C326675","24E124126C326708","24E124126C326655","24E124126C326081","24E124126C326637","24E124126C326709"]

dfID['EUI'] = np.select(conditions, values)

df_underground = dfEUI.append(dfID,ignore_index=True)
df_underground.reset_index(inplace=True, drop=True)
df_underground = df_underground.sort_values(by='DateTime')

#Add column point
conditions = [
    (df_underground['EUI'] == "24E124126C326140"),
    (df_underground['EUI'] == "24E124126C326742"),
    (df_underground['EUI'] == "24E124126C326591"),
    (df_underground['EUI'] == "24E124126C326567"),
    (df_underground['EUI'] == "24E124126C326675"),
    (df_underground['EUI'] == "24E124126C326709"),
    (df_underground['EUI'] == "24E124126C326708"),
    (df_underground['EUI'] == "24E124126C326655"),
    (df_underground['EUI'] == "24E124126C326081"),
    (df_underground['EUI'] == "24E124126C326637")
]

point = [1,2,3,4,5,6,7,8,9,10]

df_underground['point'] = np.select(conditions, point)

df_weather = weatherprep(df_weather)

#rain density
dfhourrain = df_weather[['DateTime','WeatherRain']]
def got_rain(x):
    if x > 0:
        return 1
    else:
        return 0

dfhourrain['got_rain'] = dfhourrain['WeatherRain'].apply(got_rain)
dfhourrain = filterdate(dfhourrain,start='2022-09-21 00:00',end='2022-12-31 23:00')

dfhourrain["day"] = dfhourrain['DateTime'].dt.day
dfhourrain["month"] = dfhourrain['DateTime'].dt.month
dfhourrain['hours'] = dfhourrain['DateTime'].dt.hour

groupweather=(dfhourrain.groupby(['month','day']).agg({'WeatherRain':'max','got_rain':'sum'}).reset_index())
groupweather.columns = ['month', 'day', 'maxrain','durationrain']
groupweather['raindensity'] = groupweather['maxrain']/groupweather['durationrain']
groupweather['raindensity'] = groupweather['raindensity'].fillna(0)


EUI = ['24E124126C326140']
df1 = df_underground[df_underground['EUI'].isin(EUI)]
df1 = prep(df1)
df1 = df1[["DateTime","WaterLevel"]]
df1.reset_index(inplace=True)
df1=df1.drop(['index'], axis=1)

EUI2 = ['24E124126C326742']
df2 = df_underground[df_underground['EUI'].isin(EUI2)]
df2 = prep(df2)
df2 = df2[["DateTime","WaterLevel"]]
df2.reset_index(inplace=True)
df2=df2.drop(['index'], axis=1)

EUI3 = ['24E124126C326591']
df3 = df_underground[df_underground['EUI'].isin(EUI3)]
df3 = prep(df3)
df3 = df3[["DateTime","WaterLevel"]]
df3.reset_index(inplace=True)
df3=df3.drop(['index'], axis=1)

EUI4 = ['24E124126C326567']
df4 = df_underground[df_underground['EUI'].isin(EUI4)]
df4 = prep(df4)
df4 = df4[["DateTime","WaterLevel"]]
df4.reset_index(inplace=True)
df4=df4.drop(['index'], axis=1)

EUI5 = ['24E124126C326675']
df5 = df_underground[df_underground['EUI'].isin(EUI5)]
df5 = prep(df5)
df5 = df5[["DateTime","WaterLevel"]]
df5.reset_index(inplace=True)
df5=df5.drop(['index'], axis=1)

EUI6 = ['24E124126C326709']
df6 = df_underground[df_underground['EUI'].isin(EUI6)]
df6 = prep(df6)
df6 = df6[["DateTime","WaterLevel"]]
df6.reset_index(inplace=True)
df6=df6.drop(['index'], axis=1)

EUI7 = ['24E124126C326708']
df7 = df_underground[df_underground['EUI'].isin(EUI7)]
df7 = prep(df7)
df7 = df7[["DateTime","WaterLevel"]]
df7.reset_index(inplace=True)
df7=df7.drop(['index'], axis=1)

EUI8 = ['24E124126C326655']
df8 = df_underground[df_underground['EUI'].isin(EUI8)]
df8 =prep(df8)
df8 = df8[["DateTime","WaterLevel"]]
df8.reset_index(inplace=True)
df8=df8.drop(['index'], axis=1)

EUI9 = ['24E124126C326081']
df9 = df_underground[df_underground['EUI'].isin(EUI9)]
df9 = prep(df9)
df9 = df9[["DateTime","WaterLevel"]]
df9.reset_index(inplace=True)
df9=df9.drop(['index'], axis=1)

EUI10 = ['24E124126C326637']
df10 = df_underground[df_underground['EUI'].isin(EUI10)]
df10 = prep(df10)
df10 = df10[["DateTime","WaterLevel"]]
df10.reset_index(inplace=True)
df10=df10.drop(['index'], axis=1)

df_weather["day"] = df_weather['DateTime'].dt.day
df_weather["month"] = df_weather['DateTime'].dt.month
df_weather['hours'] = df_weather['DateTime'].dt.hour

df_weatherfil = filterdate(df_weather,start='2022-09-21 00:00',end='2022-12-31 23:00')

grouped_df=(df_weatherfil.groupby(['month','day']).agg({'WeatherRain':'max'}).rename(columns={'value':'max Rain'}).reset_index())

grouped_df["month-day"] = grouped_df['month'].astype(str)+"-"+ grouped_df["day"].astype(str)

df1 = adddate(df1)
df2 = adddate(df2)
df3 = adddate(df3)
df4 = adddate(df4)
df5 = adddate(df5)
df6 = adddate(df6)
df7 = adddate(df7)
df8 = adddate(df8)
df9 = adddate(df9)
df10 = adddate(df10)


df1group=(df1.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df2group=(df2.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df3group=(df3.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df4group=(df4.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df5group=(df5.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df6group=(df6.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df7group=(df7.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df8group=(df8.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df9group=(df9.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())
df10group=(df10.groupby(['month','day']).agg({'WaterLevel':'mean'}).rename(columns={'value':'avg waterlevel'}).reset_index())

df1group["month-day"] = df1group['month'].astype(str)+"-"+ df1group["day"].astype(str)
df2group["month-day"] = df2group['month'].astype(str)+"-"+ df2group["day"].astype(str)
df3group["month-day"] = df3group['month'].astype(str)+"-"+ df3group["day"].astype(str)
df4group["month-day"] = df4group['month'].astype(str)+"-"+ df4group["day"].astype(str)
df5group["month-day"] = df5group['month'].astype(str)+"-"+ df5group["day"].astype(str)
df6group["month-day"] = df6group['month'].astype(str)+"-"+ df6group["day"].astype(str)
df7group["month-day"] = df7group['month'].astype(str)+"-"+ df7group["day"].astype(str)
df8group["month-day"] = df8group['month'].astype(str)+"-"+ df8group["day"].astype(str)
df9group["month-day"] = df9group['month'].astype(str)+"-"+ df9group["day"].astype(str)
df10group["month-day"] = df10group['month'].astype(str)+"-"+ df10group["day"].astype(str)

df1rain = pd.merge(df1group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df1rain = df1rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df2rain = pd.merge(df2group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df2rain = df2rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df3rain = pd.merge(df3group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df3rain = df3rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df4rain = pd.merge(df4group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df4rain = df4rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df5rain = pd.merge(df5group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df5rain = df5rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df6rain = pd.merge(df6group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df6rain = df6rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df7rain = pd.merge(df7group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df7rain = df7rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df8rain = pd.merge(df8group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df8rain = df8rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df9rain = pd.merge(df9group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df9rain = df9rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]
df10rain = pd.merge(df10group,grouped_df,left_on='month-day',right_on='month-day',how='right')
df10rain = df10rain[['month-day','WeatherRain','WaterLevel','lag1','lag2']]

