import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set page title
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# Title
st.title("Malaysia Condominium House Price Predictor")

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv("house_clean_data_11.csv")
    return data

data = load_data()

# Prepare features (X) and target (y)
X = data.drop(['price'], axis=1)
Y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the model
np.random.seed(42)
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=500)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display model performance
st.header("Model Performance")
# Display model performance metrics
st.write(f"R-squared Score: {r2:.2f}")
st.write(f"Mean Absolute Error: RM{mae:.2f}")

# User input for prediction
st.header("Predict Price")
st.write("Enter the following details to predict Your House Prices:")

soil_type = st.selectbox("Soil Type", options=[1, 2, 3])
rainfall = st.slider("Rainfall (mm)", min_value=0, max_value=500, value=200)
temperature = st.slider("Temperature (°C)", min_value=0, max_value=40, value=25)
previous_yield = st.number_input("Previous Yield (kg/ha)", min_value=0, value=100)

# Make prediction
if st.button("Predict Yield"):
    new_data = pd.DataFrame({
        'soil_type': [soil_type],
        'rainfall': [rainfall],
        'temperature': [temperature],
        'previous_yield': [previous_yield]
    })
    predicted_yield = model.predict(new_data)
    st.success(f"Predicted Crop Yield: {predicted_yield[0]:.2f} kg/ha")

# Display sample data
st.header("Sample Data")
st.dataframe(data)

# Add information about the app
st.sidebar.header("About")
st.sidebar.info("This is a web application demonstration for Malaysia Condo House Predictor. The model is trained on a robuts dataset taken from Kaggle that consist of 3913 data entry with 19 different Features.")