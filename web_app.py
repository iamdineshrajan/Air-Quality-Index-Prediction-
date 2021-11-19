import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/DELL/Mini Project/trained_model.sav', 'rb'))


# creating a function for Prediction

def AQI_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

    
    
  
def main():
    
    
    # giving a title
    st.title('Air Quality Index Prediction Web App')
    
    
    # getting the input data from the user
    
    
    AverageTemperature = st.text_input('Avgerage Temperature')
    MaximumTemperature = st.text_input('Maximum Temperature')
    MinimumTemperature = st.text_input('Minimum Temperature')
    AtmosphericPressure = st.text_input('Atmospheric pressure')
    AverageHumidity = st.text_input('Average relative humidity')
    AverageVisibility = st.text_input('Average visibility')
    AverageWindSpeed = st.text_input('Average wind speed')
    MaximumSustainedWindSpeed = st.text_input('Maximum sustained wind speed')
    
    
    # code for Prediction
    air_quality = ''
    
    # creating a button for Prediction
    
    if st.button('AQI Result'):
        air_quality = AQI_prediction([AverageTemperature, MaximumTemperature, MinimumTemperature,
                                      AtmosphericPressure, AverageHumidity,
                                      AverageVisibility, AverageWindSpeed, MaximumSustainedWindSpeed])
        
        
    st.success(air_quality)
    
    
    
    
    
if __name__ == '__main__':
    main()