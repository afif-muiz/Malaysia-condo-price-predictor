import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from streamlit_folium import st_folium
import folium

# Set page title
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# Title
st.title("Malaysia Condominium House Price Predictor")

# Add information about the app
st.sidebar.header("About")
st.sidebar.info("This is a web application demonstration for Malaysia Condo House Predictor. The model is trained on a robuts dataset taken from Kaggle that consist of 3913 data entry with 19 different Features.")

algorithms = st.sidebar.multiselect(
        "Select algorithms to evaluate",
        ["Linear Regression", "Random Forest Regression", "XGBoost", "Lasso Regression", "Decision Tree"]
    )

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv("database/house_clean_data_11.csv")
    return data

data = load_data()

# Display sample data
st.header("Sample Data")
st.dataframe(data)

# Prepare features (X) and target (y)
X = data.drop(['price'], axis=1)
Y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

st.write("Evaluation Results:")
results = []

for algorithm in algorithms:
    if algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "XGBoost":
        model = XGBRegressor(verbosity=0)
    elif algorithm == "Random Forest Regression":
        model = RandomForestRegressor(n_estimators=100)
    elif algorithm == "Lasso Regression":
        model = Lasso()
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        continue
        
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
        
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": algorithm,
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R2 Score": r2
    })


# Display scatter plot
    st.write(f"Scatter Plot for {algorithm}")
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Actual vs Predicted for {algorithm}")
    st.pyplot(fig)

# Display results
if results:
    results_df = pd.DataFrame(results)
    st.write(results_df)
else:
    st.write("No algorithms selected.")
    
# User input for prediction
st.header("Predict Price")
st.write("Enter the following details to predict Your House Prices:")


Bedroom = st.slider("Number of Bedroom", min_value=0, max_value=10, value=2)
Bathroom = st.slider("Number of Bathroom", min_value=0, max_value=5, value=2)
Property_Size = st.slider("Property Size", min_value=500, max_value=3000, value=1000)
Latitude = st.number_input("Latitude", min_value=0.0, max_value=90.0, format="%.4f")
Longitude = st.number_input("Longitude", min_value=0.0, max_value=360.0, format="%.4f")
Parking_Lot = st.selectbox("Parking Lot", options=[0, 1, 2])
Facility_Barbeque_area = st.selectbox("Facility Barbeque Area", options=[0,1])
Facility_Club_house = st.selectbox("Facility Club House", options=[0,1])
Facility_Gymnasium = st.selectbox("Facility Gym", options=[0,1])
Facility_Jogging_Track = st.selectbox("Facility Jogging Track", options=[0,1])
Facility_Lift = st.selectbox("Facility Lift", options=[0,1])
Facility_Minimart = st.selectbox("Facility Minimart", options=[0,1])
Facility_mph = st.selectbox("Facility Multipurpose Hall", options=[0,1])
Facility_Parking = st.selectbox("Facility Parking", options=[0,1])
Facility_Playground = st.selectbox("Facility Playground", options=[0,1])
Facility_Sauna = st.selectbox("Facility Sauna", options=[0,1])
Facility_Security = st.selectbox("Facility Security", options=[0,1])
Facility_Squash_Court = st.selectbox("Facility Squash Court", options=[0,1])
Facility_Swimming_pool = st.selectbox("Facility Swimming Pool", options=[0,1])
Facility_Tennis_court = st.selectbox("Facility Tennis Court", options=[0,1])

# Make prediction
if st.button("Predict Price"):
    new_data = pd.DataFrame({
        'Bedroom': [Bedroom],
        'Bathroom': [Bathroom],
        'Property Size': [Property_Size],
        'Latitude' : [Latitude],
        'Longitude' : [Longitude],
        'Parking Lot': [Parking_Lot], 
        'Facility_Barbeque area': [Facility_Barbeque_area],
        'Facility_Club house': [Facility_Club_house],
        'Facility_Gymnasium': [Facility_Gymnasium],
        'Facility_Jogging Track': [Facility_Jogging_Track],
        'Facility_Lift': [Facility_Lift],
        'Facility_Minimart' : [Facility_Minimart],
        'Facility_Multipurpose hall' : [Facility_mph],
        'Facility_Parking' : [Facility_Parking],
        'Facility_Playground' : [Facility_Playground],
        'Facility_Sauna' : [Facility_Sauna],
        'Facility_Security' : [Facility_Security],
        'Facility_Squash Court' : [Facility_Squash_Court],
        'Facility_Swimming Pool' : [Facility_Swimming_pool],
        'Facility_Tennis Court' : [Facility_Tennis_court]
    })
    predicted_price = model.predict(new_data)
    st.success(f"Predicted Condomonimum Prices: RM {predicted_price[0]:.2f}")
    
