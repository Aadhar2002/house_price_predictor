#Import libraries
import joblib
import pandas as pd

#Load saved rent model artifacts
def load_rent_artifacts():
    model = joblib.load("models/rent_model.pkl")
    feature_columns = joblib.load("models/rent_feature_columns.pkl")
    return model, feature_columns

#Prepare input row for prediction
def prepare_rent_input(location, sqft, bath, bhk, feature_columns):
    input_data = {col: 0 for col in feature_columns}

    if "sqft" in input_data:
        input_data["sqft"] = sqft

    if "bath" in input_data:
        input_data["bath"] = bath

    if "bhk" in input_data:
        input_data["bhk"] = bhk

    location_column = f"location_{str(location).strip().lower()}"
    if location_column in input_data:
        input_data[location_column] = 1



    input_df = pd.DataFrame([input_data])
    return input_df

#Predict rent
def predict_rent(location, sqft, bath, bhk):
    model, feature_columns = load_rent_artifacts()

    input_df = prepare_rent_input(
        location=location,
        sqft=sqft,
        bath=bath,
        bhk=bhk,
        feature_columns=feature_columns
    )

    prediction = model.predict(input_df)[0]
    return float(prediction)

#Test prediction
if __name__ == "__main__":
    predicted_rent = predict_rent(
        location="hebbal",
        sqft=1200,
        bath=2,
        bhk=2
    )

    print(f"Predicted Rent: ₹{predicted_rent:.2f}")