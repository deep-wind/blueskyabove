import os
import json
import gdown
import requests
import numpy as np
import pandas as pd
import folium
import datetime
from netCDF4 import Dataset
from sentinelsat import SentinelAPI, geojson_to_wkt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import folium_static, st_folium


# Constants
LAT_MIN, LAT_MAX = 51.25, 51.75
LNG_MIN, LNG_MAX = -0.6, 0.28
FILE_ID = '1RVnGXF9RMYb4qgfHFawlASp1dtZaYwwM'
LOCAL_FILENAME = '5days_combined.nc'

# Download and Load Dataset
def load_dataset():
    if not os.path.exists(LOCAL_FILENAME):
        gdown.download(f"https://drive.google.com/uc?export=download&id={FILE_ID}", LOCAL_FILENAME, quiet=True)
    return Dataset(LOCAL_FILENAME)

L3_data1 = load_dataset()
LAT = L3_data1.variables['latitude'][:]
LON = L3_data1.variables['longitude'][:]
NO2 = L3_data1.variables['tropospheric_NO2_column_number_density'][:,:,:]

# Prediction Function
def predict(latitude_input, longitude_input, date):
    predict_days = min(abs((datetime.datetime.strptime('2024-09-05', "%Y-%m-%d") - datetime.datetime.strptime(date, "%Y-%m-%d")).days), 4)
    st.write("Prediction Days:", predict_days)

    # Find nearest latitude and longitude indices
    lat_idx = np.argmin((LAT - latitude_input) ** 2)
    lon_idx = np.argmin((LON - longitude_input) ** 2)

    # Create DataFrame for prediction range
    start_date = L3_data1.variables['time'].units[14:24]
    date_range = pd.date_range(start=start_date, periods=NO2.shape[0])
    df = pd.DataFrame({"NO2": NO2[:, lat_idx, lon_idx]}, index=date_range)
    df.fillna(0, inplace=True)

    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['NO2'].values.reshape(-1, 1))
    train_size = int(len(scaled_data) * 0.70)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            dataX.append(dataset[i:(i + time_step), 0])
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 1
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, verbose=1)

    # Generate predictions
    x_input = test_data[-1:].reshape(1, -1)
    temp_input = list(x_input[0])
    lst_output = []
    for _ in range(predict_days):
        x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())

    # Prepare output DataFrame
    no2_output = pd.DataFrame(scaler.inverse_transform(lst_output), columns=['NO2 Concentration (mol/m²)'])
    no2_output['Date'] = pd.date_range(start=date, periods=predict_days).date
    st.write(no2_output.style.background_gradient(cmap='YlOrRd', subset=['NO2 Concentration (mol/m²)']))
    return round(no2_output.iloc[-1]['NO2 Concentration (mol/m²)'], 4)

# Main Function
def main():
    st.title("NO₂ Prediction")
    st.markdown("### Choose a Location on the Map 📌")
    
    # Create map
    map_center = [(LAT_MIN + LAT_MAX) / 2, (LNG_MIN + LNG_MAX) / 2]
    m = folium.Map(location=map_center, zoom_start=8)
    folium.Rectangle(bounds=[[LAT_MIN, LNG_MIN], [LAT_MAX, LNG_MAX]], color='blue', fill=True, fill_opacity=0.1).add_to(m)
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=400, width=700)

    # Handle map clicks
    try:
        latitude_input = float(map_data['last_clicked']['lat'])
        longitude_input = float(map_data['last_clicked']['lng'])
        st.write(f"Selected Latitude: {latitude_input}, Longitude: {longitude_input}")
    except:
        st.warning("No location selected. Please click on the map.")

    # Date input
    date = st.date_input('Date', value=datetime.date(2024, 9, 6), min_value=datetime.date(2024, 9, 6), max_value=datetime.date(2025, 9, 6))

    # Predict button
    if st.button("Predict"):
        output = predict(latitude_input, longitude_input, str(date))
        st.success(f"Predicted NO₂ Level: {output} mol/m²")

# Run the app
if __name__ == '__main__':
    main()
