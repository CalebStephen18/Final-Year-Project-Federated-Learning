import streamlit as st
import requests
import sys
import json
import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
df1 = pd.read_csv("datasetz/bang1.csv")
df2 = pd.read_csv("datasetz/bom1.csv")
df3 = pd.read_csv("datasetz/cbe1.csv")
df4 = pd.read_csv("datasetz/hyd1.csv")
df5 = pd.read_csv("datasetz/maa1.csv")
df6 = pd.read_csv("datasetz/mad1.csv")

def calculate_error_percentage(actual, predicted):
    error = abs(actual - predicted)
    error_percentage = (error / actual) * 100
    return error_percentage

# Function to fetch temperature and humidity data from API
def fetch_weather_data(place, year, month, day):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{place}/{year}-{month}-{day}/{year}-{month}-{day}?unitGroup=metric&key=ZQ7EFG9997MA5APLB7Z4YCSC6&contentType=json"
    response = requests.get(url)
    if response.status_code != 200:
        st.error('Unexpected Status code: ' + str(response.status_code))
        st.stop()
    return response.json()

# Function to calculate heat index
def calculate_heat_index(temp, humidity):
    c1 = -8.78469475556
    c2 = 1.61139411
    c3 = 2.33854883889
    c4 = -0.14611605
    c5 = -0.012308094
    c6 = -0.0164248277778
    c7 = 0.002211732
    c8 = 0.00072546
    c9 = -0.000003582

    heat_index = (c1 + (c2 * temp) + (c3 * humidity) +
                  (c4 * temp * humidity) +
                  (c5 * temp ** 2) +
                  (c6 * humidity ** 2) +
                  (c7 * temp ** 2 * humidity) +
                  (c8 * temp * humidity ** 2) +
                  (c9 * temp ** 2 * humidity ** 2))
    return heat_index

# Streamlit UI
st.title('Temperature Prediction Web App')

# User input fields
# User input field as dropdown menu
place_options = ["Bangalore", "Mumbai", "Coimbatore", "Hyderabad", "Chennai", "Madurai"]
place = st.selectbox('Place', place_options, index=1)  
year = st.number_input('Year', min_value=2000, max_value=2025, step=1)
month = st.number_input('Month', min_value=1, max_value=12, step=1)
day = st.number_input('Day', min_value=1, max_value=31, step=1)
hour = st.number_input('Hour', min_value=0, max_value=23, step=1)

# Fetch weather data from API

def calculate_energy_cost(current_temp, comfortable_temp):
    delta_temp = abs(comfortable_temp - current_temp) / 3600000
    energy = 1 * 4.18 * delta_temp
    if energy <= 100:
        cost = 0
    elif energy <= 200:
        cost = 5.80 * energy
    else:
        cost = 6.50 * energy
    return energy, cost

# Extract temperature and humidity based on user input hour
def get_temp_humidity_at_time(data, hour):
    for day_data in data['days']:
        for hour_data in day_data['hours']:
            if hour_data['datetime'].endswith(str(hour) + ":00:00"):
                return hour_data['temp'], hour_data['humidity']
    return None, None

# Make prediction
if st.button('Predict'):
    jsonData = fetch_weather_data(place, year, month, day)
    # Get temperature and humidity at user input hour
    temp, humidity = get_temp_humidity_at_time(jsonData, hour)

    # Display temperature and humidity
    if temp is not None and humidity is not None:
        st.write(f"Current Temperature: {temp}°C, Current Humidity: {humidity}%")
    else:
        st.write("Data not available for the given hour.")

    # Load the model
    model = load_model('model_results_/custom_covnet_107.h5')
    if temp is not None and humidity is not None:
        # Get the appropriate DataFrame based on the selected place
        if place == "Bangalore":
            df = df1
        elif place == "Mumbai":
            df = df2
        elif place == "Coimbatore":
            df = df3
        elif place == "Hyderabad":
            df = df4
        elif place == "Chennai":
            df = df5
        elif place == "Madurai":
            df = df6
        # Add other conditions for different locations as needed

        # Filter DataFrame for the specified hour, day, and month
        filtered_df = df[(df['month'] == int(month)) & (df['day'] == int(day)) & (df['hour'] == int(hour))]

        heat_index = calculate_heat_index(temp,humidity)
        prediction = model.predict(np.array([[year, month, day, hour, temp, humidity, heat_index]], dtype=np.float32))[0][0]
        st.write(f'Predicted Comfortable Temperature: {prediction:.2f}°C')

        average_simulated_temperature = filtered_df['Simulated_Comfort_Temperature'].mean()
        st.write(f'Average Simulated Temperature based on user habits: {average_simulated_temperature:.2f}°C')

        # Calculate error percentage
        error_percentage = calculate_error_percentage(average_simulated_temperature, prediction)
        st.write(f'Error Percentage in Prediction for the requested hour: {error_percentage:.2f}%')
        
        # Collect temperature and humidity data for each hour starting from the user-selected hour
        data = []
        for hour_data in jsonData['days'][0]['hours']:
            data_hour = int(hour_data['datetime'].split(':')[0])  # Extract hour from datetime string
            if data_hour >= hour:
                temp, hum = hour_data['temp'], hour_data['humidity']
                heat_index = calculate_heat_index(temp, hum)
                data.append([year, month, day, data_hour, temp, hum, heat_index])

        # Make prediction using the loaded model for all the collected data
        predictions = model.predict(np.array(data, dtype=np.float32)).flatten()

        chart_data = pd.DataFrame({'Hour': [item[3] for item in data],
                                   'Existing Temperature': [item[4] for item in data],
                                   'Estimated Comfortable Temperature': predictions})

        # Plotting temperatures using st.line_chart
        st.line_chart(chart_data.set_index('Hour'), color=["#fd0", "#f0f"])


        energy_consumption, cost = calculate_energy_cost(temp, prediction)
        st.write(f'Energy Consumption to reach comfortable temperature: {energy_consumption:.2f} kWh')
        st.write(f'Cost associated in rupees: {cost:.2f}')