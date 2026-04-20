#Import libraries
import joblib
import pandas as pd

#Load artifacts
def load_artifacts():
    """
    Load saved model and feature columns.
    """
    model = joblib.load("models/best_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    location_avg_mapping = joblib.load("models/location_avg_price.pkl")
    return model, feature_columns, location_avg_mapping

#Prepare input
def prepare_input(location, total_sqft, bath, bedroom, feature_columns, location_avg_mapping):
    """
    Prepare one row of model input using the saved training columns.
    """
    input_data = {col: 0 for col in feature_columns}

    if "total_sqft" in input_data:
        input_data["total_sqft"] = total_sqft

    if "bath" in input_data:
        input_data["bath"] = bath

    if "bedroom" in input_data:
        input_data["bedroom"] = bedroom

    #if "sqft_per_bedroom" in input_data:
        #input_data["sqft_per_bedroom"] = total_sqft/bedroom

    if "location_avg_price" in input_data:
        input_data["location_avg_price"] = location_avg_mapping.get(location, 0)

    location_column = f"location_{location}"
    if location_column in input_data:
        input_data[location_column] = 1

    input_df = pd.DataFrame([input_data])

    return input_df

#Predict Price
def predict_price(location, total_sqft, bath, bedroom):
    """
    Predict house price using the saved best model.
    """
    model, feature_columns, location_avg_mapping = load_artifacts()

    input_df = prepare_input(
        location=location,
        total_sqft=total_sqft,
        bath=bath,
        bedroom=bedroom,
        feature_columns=feature_columns,
        location_avg_mapping= location_avg_mapping
    )

    prediction = model.predict(input_df)[0]
    return prediction

if __name__ == "__main__":
    predicted_price = predict_price(
        location="Whitefield",
        total_sqft = 1200,
        bath = 2,
        bedroom = 2
    )

    print(f"Predicted Price: {predicted_price:.2f} Lakhs")
