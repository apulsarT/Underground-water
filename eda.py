import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates

df_underground = pd.read_csv('data/underground.csv')
df_weather = pd.read_csv('data/weather.csv')

#Functions
def prep(df):
  df['DateTime']=pd.to_datetime(df['DateTime'])
  df['DateTime'] = df['DateTime'] + timedelta(hours=8)
  df = df[df['DateTime'].dt.year > 2021]
  df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:00')
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  df['month'] = df['DateTime'].dt.month
  df['day'] = df['DateTime'].dt.day
  df['hours'] = df['DateTime'].dt.hour
  df['minutes'] = df['DateTime'].dt.minute
  df['DateTime'] = df['DateTime'].dt.round('10min')
  df = df.groupby(['month','day','hours','minutes']).first().reset_index()
  df= df.sort_values(by='DateTime')
  return df 

def weatherprep(df):
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  # Weather data is recorded in different time zones, all data needs to be added for 8 hours to change to the correct time zone
  df['DateTime'] = df['DateTime'] + timedelta(hours=8)
  df['month'] = df['DateTime'].dt.month
  df['day'] = df['DateTime'].dt.day
  df['hours'] = df['DateTime'].dt.hour
  df['minutes'] = df['DateTime'].dt.minute
  df['date'] = df['DateTime'].dt.date  
  df['DateTime'] = df['DateTime'].dt.round('10min')
  df = df.groupby(['month','day','hours','minutes']).first().reset_index()
  df = df.groupby(['month','day','hours']).first().reset_index()
  df = df[['DateTime','WeatherRain']]
  df = df[df['DateTime'].dt.year > 2021] 
  #df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:00:00')
  df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:00')
  df['DateTime'] = pd.to_datetime(df['DateTime'])
  df.drop_duplicates(keep='first', inplace=True) 
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


# For November data, the EUI changes to DeviceID column. Change the DeviceID to their respective EUI.
df1 = df_underground[df_underground['EUI'].notna()]
df2 = df_underground[df_underground['DeviceID'].notna()]

conditions = [
    (df2['DeviceID'] == "UDW-001"),
    (df2['DeviceID'] == "UDW-002"),
    (df2['DeviceID'] == "UDW-003"),
    (df2['DeviceID'] == "UDW-004"),
    (df2['DeviceID'] == "UDW-005"),
    (df2['DeviceID'] == "UDW-006"),
    (df2['DeviceID'] == "UDW-007"),
    (df2['DeviceID'] == "UDW-009"),
    (df2['DeviceID'] == "UDW-010"),
    (df2['DeviceID'] == "UDW-011")
]

values = ["24E124126C326140","24E124126C326742","24E124126C326591","24E124126C326567","24E124126C326675","24E124126C326708","24E124126C326655","24E124126C326081","24E124126C326637","24E124126C326709"]
df2['EUI'] = np.select(conditions, values)

df_underground = df1.append(df2,ignore_index=True)
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

#pre-process the data
df_weather = weatherprep(df_weather)
#df_underground = prep(df_underground) 

#extract each sensor(point) into each dataframe table.
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



#Heatmap 
df_underground = df_underground[['EUI','DateTime','WaterLevel']]

#Add latitude and longitude columns.
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

values = [2.8891,2.8859,2.891,2.8903,2.8989,2.8951,2.8837,2.901,2.8947,2.8801]
values2 = [101.36,101.36,101.37,101.37,101.37,101.36,101.37,101.36,101.35,101.35]

df_underground['latitude'] = np.select(conditions, values)
df_underground['longitude'] = np.select(conditions,values2)






df_weather_sepoct = filterdate(df_weather,start='2022-09-01 00:00',end='2022-10-30 23:00')

