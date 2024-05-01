import streamlit as st
import pandas as pd
import pickle as pk

# Load  models and scalers outside the functions
one_hot_encoder = pk.load(open('car_one_hot_encoder.pkl', 'rb'))
binary_encoder = pk.load(open('car_binary_encoder.pkl', 'rb'))
robust_scaler = pk.load(open('car_robust_scaler.pkl', 'rb'))
rf_model = pk.load(open('random_forest_model.pkl', "rb"))
xgb_model = pk.load(open('xgboost_model.pkl', "rb"))
cat_model = pk.load(open('catboost_model.pkl', "rb"))

# Fuction
def preprocessing (input_data):
    # Getting brand_names from name
    input_data["brand_name"] = input_data["name"].apply(lambda x: x.split()[0])
    
    # Dropping the name column
    input_data = input_data.drop("name", axis = 1)

    # Feature Engineering to create engine_power column from torque and rpm
    def eng_power(torque_value, rpm_value):
        return (torque_value * rpm_value) / 7127 # 7127 is the conversion factor for Nm and rpm to hp.

    # Applying the function to each row in the DataFrame and creating the new engine_power_hp column
    input_data["engine_power"] = input_data.apply(lambda row: eng_power(row["torque_value"], row["rpm_value"]), axis = 1)

    # Dropping torque and rpm values
    input_data = input_data.drop(["torque_value", "rpm_value"], axis = 1)

    # Load Encoders
    one_hot_encoder = pk.load(open('car_one_hot_encoder.pkl', 'rb'))
    binary_encoder = pk.load(open('car_binary_encoder.pkl', 'rb'))

    # Encoding the data
    input_data_hot = one_hot_encoder.transform(input_data)
    input_data_bin = binary_encoder.transform(input_data_hot)

    # Scaling the data
    robust_scaler = pk.load(open('car_robust_scaler.pkl', 'rb'))
    input_data_scaled = pd.DataFrame(robust_scaler.transform(input_data_bin), columns = input_data_bin.columns)

    return input_data_scaled

# Function predicting the price using the preprocessed data
def predict_price(processed_data):
    # Load models
    rf = pk.load(open('random_forest_model.pkl', "rb"))
    xgb = pk.load(open('xgboost_model.pkl', "rb"))
    cat = pk.load(open('catboost_model.pkl', "rb"))
    
    # Make predictions and compute weighted average
    rf_pred = rf.predict(processed_data)
    cat_pred = cat.predict(processed_data)
    xgb_pred = xgb.predict(processed_data)
    
    final_prediction = ((1 * rf_pred) +
                        (2 * xgb_pred) +
                        (3 * cat_pred))
    
    return final_prediction / (1+2+3)


# Function for deployment
def car_price_prediction(raw_data):
    # Preprocess the input data
    preprocessed_data = preprocessing(raw_data)
    
    # Get the prediction price
    price_prediction = predict_price(preprocessed_data)
    
    return price_prediction

# Streamlit app
st.title('Car Price Prediction App')

# Input fields for the user to enter car details
name = st.text_input('Car Name')
year = st.number_input('Year', min_value=1900, max_value=2024, step=1)
km_driven = st.number_input('Kilometers Driven', min_value=0, step=1000)
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input('Mileage (kmpl)', min_value=0.0, format="%.2f")
engine = st.number_input('Engine (CC)', min_value=0)
max_power = st.number_input('Max Power (bhp)', min_value=0.0, format="%.2f")
seats = st.number_input('Seats', min_value=1, max_value=10, step=1)
torque_value = st.number_input('Torque (Nm)', min_value=0.0, format="%.2f")
rpm_value = st.number_input('RPM', min_value=0)

# Button to make prediction
if st.button('Predict Price'):
    # Creating a DataFrame with the input data
    input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, torque_value, rpm_value]],
                                  columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'torque_value', 'rpm_value'])
        
    # Getting the prediction
    prediction = car_price_prediction(input_data)
        
    # Displaying the prediction
    st.success(f'The predicted price of the car is: ${prediction[0]:,.2f}')