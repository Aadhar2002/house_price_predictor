#Import Libraries
import streamlit as st
import joblib
from src.predict import predict_price
from src.predict_rent import predict_rent

st.set_page_config(page_title = "House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor")
st.write("Estimate house prices for buying or renting in Bangalore.")
mode = st.selectbox("Select Mode", ["Buy", "Rent"])

@st.cache_data
#load Locations
def load_locations():
    feature_columns = joblib.load("models/feature_columns.pkl")
    locations = []

    for col in feature_columns:
        if col.startswith("location_"):
            clean_name = col.replace("location_", "", 1)
            locations.append(clean_name)

    locations = sorted(locations)
    return locations

locations = load_locations()

location = st.selectbox("Location", locations)
total_sqft = st.number_input("Total Square Feet", min_value=100.0, value=1200.0, step=50.0)
bedroom = st.number_input("Bedrooms", min_value=1, value=2, step=1)
bath = st.number_input("Bathrooms", min_value=1, value=2, step=1)

if mode == "Rent":
    st.info(
        """Rent predictions are strongest for locations present in the Bangalore rent dataset."""
        """Some uncovered areas may use fallback estimation."""
    )

if st.button("Predicted Price"):
    if mode == "Buy":
        predicted_price = predict_price(location, total_sqft, bath, bedroom)
        st.success(f"Predicted Price: ₹{predicted_price:.2f} Lakhs")

        predicted_rent = predict_rent(location, total_sqft, bath, bedroom)
        st.success(f"Predicted Rent: ₹{predicted_rent:.2f} per month")