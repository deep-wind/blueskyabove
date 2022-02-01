# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 19:18:12 2022

@author: PRAMILA
"""

import streamlit as st
import pandas as pd
#import harp
from netCDF4 import Dataset
import os
from os.path import join
import glob
import xarray as xr
import numpy as np
from numpy import array
import itertools
import datetime
import time
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt 
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import base64

# BlueSky Above: Pollution estimation using hyper-spectral satellite imagery and maps



# Install a pip package in the current Jupyter kernel
import sys
import smopy
#WMF Maps terms of use https://foundation.wikimedia.org/wiki/Maps_Terms_of_Use
#OSM tile server terms of use/Tile usage policy https://operations.osmfoundation.org/policies/tiles/

LatMin=51.25
LatMax=51.75
LngMin=-0.6
LngMax=0.28

# #Map with labels
# map = smopy.Map((LatMin,LngMin, LatMax,LngMax))
# map.save_png('London.png')
# #map.show_ipython()

# #Map without labels (if needed)
# map = smopy.Map((LatMin,LngMin, LatMax,LngMax), tileserver="https://tiles.wmflabs.org/osm-no-labels/{z}/{x}/{y}.png")#,tilesize=512,maxtiles=9)
# map.save_png('London2.png')
# #map.show_ipython()

#A matplotlib figure can be created from these images which could be used for further processing. Further downsizing or cropping or overlays can be done as per your need.
# The Map object comes with a to_pixels method to convert from geographical coordinates to pixels in this image.
#https://github.com/rossant/smopy

#x, y = map.to_pixels(51.5072, -0.1276)
#ax = map.show_mpl(figsize=(8, 8))
#ax.plot(x, y, 'or', ms=10, mew=2)



# from sentinelsat import SentinelAPI, geojson_to_wkt
# import json
# #long then lat here
# geojsonstring='{{"type":"FeatureCollection","features":[{{"type":"Feature","properties":{{}},"geometry":{{"type":"Polygon","coordinates":[[[{LongiMin},{LatiMin}],[{LongiMax},{LatiMin}],[{LongiMax},{LatiMax}],[{LongiMin},{LatiMax}],[{LongiMin},{LatiMin}]]]}}}}]}}'.format(LongiMin=LngMin,LatiMin=LatMin,LongiMax=LngMax,LatiMax=LatMax)

# #username and password 's5pguest' datahubs url
# api = SentinelAPI('s5pguest', 's5pguest' , 'https://s5phub.copernicus.eu/dhus')
# footprint = geojson_to_wkt(json.loads(geojsonstring))

# startdate='20210325'
# enddate='20210405'
# #L2: registered
# products_to_download = api.query(footprint, date = (startdate,enddate),
#                       producttype = 'L2__NO2___' )

# api.download_all(products_to_download, directory_path='data\L2new')





import urllib
import os
#01-03-2021 to 30-04-2021 https://uk-air.defra.gov.uk/data/

#If you just have the ipynb file you can download the data using the script below. If you have cloned the repo then the sample data files are already there in the GroundData folder.

#try: 
#     os.mkdir("GroundData") 
# except OSError as error: 
#     print(error)  

# #Data selector example  https://uk-air.defra.gov.uk/data/data_selector_service? 
# urllib.request.urlretrieve('https://raw.githubusercontent.com/williamnavaraj/BlueSkyChallenge/main/GroundData/Data_Selector_Example_Defra.pdf', 'GroundData/Data_Selector_Example_Defra.pdf')

# #London Bloomsbury https://uk-air.defra.gov.uk/data/show-datavis?q=2862199&type=auto
# urllib.request.urlretrieve('https://raw.githubusercontent.com/williamnavaraj/BlueSkyChallenge/main/GroundData/LondonBloomsbury.csv', 'GroundData/LondonBloomsbury.csv')

# #London Hillingdon https://uk-air.defra.gov.uk/data/show-datavis?q=2862385&type=auto
# urllib.request.urlretrieve('https://raw.githubusercontent.com/williamnavaraj/BlueSkyChallenge/main/GroundData/LondonHillington.csv', 'GroundData/LondonHillington.csv')

# #London Bexley https://uk-air.defra.gov.uk/data/show-datavis?q=2862404&type=auto
# urllib.request.urlretrieve('https://raw.githubusercontent.com/williamnavaraj/BlueSkyChallenge/main/GroundData/LondonBexley.csv', 'GroundData/LondonBexley.csv')




import time
# https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %.3f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
#Save your compressed information data from the satellite data as a file
#os.path.getsize("/path_to_your_file") #This file size is one of the evaluation criteria
#Time taken for your regression calculation will be tested using the tic toc routines


st.set_page_config(
page_title=" No2 Prediction ➼",
page_icon="🚩",
initial_sidebar_state="expanded"
)



def predict(latitude_input,longitude_input,date):
    
    
    
    ##############################################
    #       COVERTION OF LEVEL-2 TO LEVEL-3      #
    ##############################################

    # input_path="C:/Users/PRAMILA/.spyder-py3/project/data/L2/"
    # export_path="C:/Users/PRAMILA/.spyder-py3/project/data/"
    
    # list_files=sorted(os.listdir(input_path))
    
    # files_inputs= sorted(list(glob.glob(join(input_path, 'S5P_OFFL_*.nc'))))
    # #print(files_inputs)
    # for file in list_files:
    #     if file.startswith("S5P_OFFL_")==False:
    #         list_files.remove(file)
            
    # for i in range(len(files_inputs)):
        
    #     product_name = harp.import_product(files_inputs[i], operations="tropospheric_NO2_column_number_density_validity>50;derive(tropospheric_NO2_column_number_density [Pmolec/cm2]);derive(datetime_stop {time}); latitude > 51.25 [degree_north]; latitude < 51.75 [degree_north];bin_spatial(50 ,51.25 ,0.01 ,88 ,-0.6 ,0.01);squash(time, (latitude_bounds,longitude_bounds));derive(latitude {latitude});derive(longitude {longitude});\
    # keep(latitude,longitude,tropospheric_NO2_column_number_density);exclude(latitude_bounds,longitude_bounds,count,weight)")
    # #export_folder="C:/Users/PRAMILA/.spyder-py3/project/data/L2/S5P_OFFL_L2__NO2____20210325T104955_20210325T123125_17863_01_010400_20210327T044257.nc".format(export_path="C:/Users/PRAMILA/.spyder-py3/project/data/L2/S5P_OFFL_L2__NO2____20210325T104955_20210325T123125_17863_01_010400_20210327T044257.nc",name=export_path.split('/')[-1].replace('L2','L3'))                                      
    
    #     harp.export_product(product_name,"{export_path}/{name}".format(export_path=export_path,name=files_inputs[i].split('/')[-1].replace('L2','L3')),file_format='netcdf')
    #     print(os.path.basename(files_inputs[i]),"done preprocessing")
    
    # attributes={os.path.basename(i):
    #                 {
    #                     'time_coverage_start': xr.open_dataset(i).attrs['time_coverage_start'],
    #                     'time_coverage_end': xr.open_dataset(i).attrs['time_coverage_end'],                   
    #               }for i in files_inputs
    #                 }
    
    # #print(dict(itertools.islice(attributes.items(),1)))
    # export_path1="C:/Users/PRAMILA/.spyder-py3/project/data/L3"
    # files_L3= sorted(list(glob.glob(join(export_path1, 'S5P_OFFL_*.nc'))))
    
    # # #print(files_L3)
    # def preprocess(ds):
    #     print('\nsource_product\n',ds.attrs['source_product'])
    #     ds['time']=pd.to_datetime(np.array([attributes[ds.attrs['source_product']]['time_coverage_start']])).values
    #     return ds
    

    # L3_data=xr.open_mfdataset(files_L3,decode_times=False,combine='nested',concat_dim='time',preprocess=preprocess,chunks={'time':100})
    # #print(L3_data)
    
    # L3_data.to_netcdf(path=r"C:\Users\PRAMILA\.spyder-py3\project\data\20days_combined.nc")
    L3_data1=Dataset("20days_combined.nc")
    #print(L3_data1.variables.keys())
    lat=L3_data1.variables['latitude'][:]
    lon=L3_data1.variables['longitude'][:]
    time_data=L3_data1.variables['time'][:]
    no2=L3_data1.variables['tropospheric_NO2_column_number_density'][0:,:,:]
    
    predict_days=abs((datetime.strptime('2021-04-13',"%Y-%m-%d")-datetime.strptime(date,"%Y-%m-%d")).days)
    #st.write("--------predict days-----:",predict_days)
    
    
    #
    sq_diff_lat=(lat-latitude_input)**2
    sq_diff_lon=(lon-longitude_input)**2
    
    min_index_lat=sq_diff_lat.argmin()
    min_index_lon=sq_diff_lon.argmin()
    
    start_date=L3_data1.variables['time'].units[14:24]
    end_date=L3_data1.variables['time'].units[14:18]+'-04-13'
    
    date_range=pd.date_range(start=start_date,end=end_date)
    
    df=pd.DataFrame(0,columns=['NO2'],index=date_range)
    dt=np.arange(0,L3_data1.variables['time'].size)
    
    for i in dt:
        df.iloc[i]=no2[i,min_index_lat,min_index_lon]
        
    df.to_csv("20days_combined.csv")
    print(os.path.getsize("20days_combined.csv"))
    
    ##############################################
    #            PREDICTION MODULE               #
    ##############################################
    ### Data Collection
    data_frame=pd.read_csv("20days_combined.csv")
    df1=data_frame.reset_index()['NO2']

    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 

    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    
    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.55)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    
    
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
     	dataX, dataY = [], []
     	for i in range(len(dataset)-time_step-1):         
        		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        		dataX.append(a)
        		dataY.append(dataset[i + time_step, 0])
     	return np.array(dataX), np.array(dataY)
    
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 7
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # print(X_train.shape), print(y_train.shape)
    # print(X_test.shape), print(ytest.shape)
  
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    ### Create the Stacked LSTM model
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(7,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    
    
    model.summary()
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=25,batch_size=2,verbose=1)
    

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    
    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    
    ### Calculate RMSE performance metrics
    
    math.sqrt(mean_squared_error(y_train,train_predict))
    
    ### Test Data RMSE
    math.sqrt(mean_squared_error(ytest,test_predict))
    
    ### Plotting 
    # shift train predictions for plotting
    look_back=7
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
      
    x_input=test_data[len(test_data)-7:].reshape(1,-1)
    
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    # demonstrate prediction for next days
    
    lst_output=[]
    n_steps=7
    i=0
    while(i<predict_days):
        
        if(len(temp_input)>7):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
    
    print(lst_output)
        
   # st.write(df3)
    no2_output=pd.DataFrame(scaler.inverse_transform(lst_output),columns=['NO2 Concentration 🏭'])
    st.write(no2_output)
    output= (no2_output.at[predict_days-1,'NO2 Concentration 🏭'])
    return output


def main():
    
    st.markdown("<h1 style ='color:green; text_align:center;font-family:times new roman;font-weight: bold;font-size:20pt;'>NO2 PREDICTION </h1>", unsafe_allow_html=True)  
    st.markdown("<h1 style='text-align: left; font-weight:bold;color:black;background-color:white;font-size:11pt;'> Enter the Location Details 🧭 </h1>",unsafe_allow_html=True)
    col1,col2 = st.columns(2)         
    with col1:
         latitude_input=st.text_input('📍 Latitude (°N)')                         
    with col2:   
         longitude_input=st.text_input('📍 Longitude (°E)')                         
    st.markdown("<h1 style='text-align: left; font-weight:bold;color:black;background-color:white;font-size:11pt;'> Enter the Timing details ⌛</h1>",unsafe_allow_html=True)
    #st.date_input('Date', value=None, min_value= value = pd.to_datetime('2010-01-01'), max_value=datetime(2030, 1, 1), key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
    date = st.date_input('📅 Date', value = pd.to_datetime('2021-04-14'),min_value= pd.to_datetime('2021-04-14'),max_value= pd.to_datetime('2021-04-30'))
    
    if st.button("Predict"):
        latitude_input=float(latitude_input)
        longitude_input=float(longitude_input)
        df_map = pd.DataFrame(
         np.random.randn(1000, 2) / [50, 50] + [latitude_input,longitude_input],
         columns=['lat', 'lon'])
        st.markdown("<h1 style='text-align: left; font-weight:bold;color:black;background-color:white;font-size:11pt;'> Selected Location </h1>",unsafe_allow_html=True)
    
        st.map(df_map)
        date=str(date)
        with st.spinner("Predicting the results...."):
             result = predict(latitude_input,longitude_input,date)
 
        st.success('Predicted NO2 Concentration is {} molecules/cm2'.format(round(result,4))) 
        st.balloons() 


if __name__ == "__main__":
    main()
