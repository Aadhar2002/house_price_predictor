#Import Libraries
import streamlit as st
from src.predict import predict_price
import joblib

st.set_page_config(page_title = "House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor")
st.write("Enter house details to estimate the price.")

@st.cache_data
#load Locations
def load_locations():
    feature_columns = joblib.load("models/feature_columns.pkl")
    locations = []

    for col in feature_columns:
        if col.startswith("location_"):
            locations.append(col.replace("location_", ""))

    locations = sorted(locations)
    return locations

locations = load_locations()

location = st.selectbox("Location", locations)
total_sqft = st.number_input("Total Square Feet", min_value=300.0, value=1200.0, step=50.0)
bedroom = st.number_input("Bedrooms", min_value=1, value=2, step=1)
bath = st.number_input("Bathrooms", min_value=1, value=2, step=1)
if st.button("Predicted Price"):
    predicted_price = predict_price(
        location=location,
        total_sqft=total_sqft,
        bath=bath,
        bedroom=bedroom
    )
    st.success(f"Predicted Price: ₹{predicted_price:.2f} Lakhs")