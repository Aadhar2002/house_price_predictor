#Import libraries
import pandas as pd
from pathlib import Path

#City filter function
def filter_city_data(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    df = df.copy()
    df["city"] = df["city"].astype(str).str.strip().str.lower()
    city_name = city_name.strip().lower()
    return df[df["city"] == city_name].copy()

#Preprocess rent data
def preprocess_rent_data(file_path: str, city_name: str = "Bangalore") -> pd.DataFrame:
    df = pd.read_csv(file_path)

    #Filter Bangalore
    df = filter_city_data(df, city_name)

    #Rename columns (align with buy model)
    df = df.rename(columns={
        "locality": "location",
        "area": "sqft",
        "bathrooms": "bath",
        "beds": "bhk",
        "rent": "price"
    })

    #Keep only required columns
    df = df[["location", "sqft", "bath", "bhk", "price"]].copy()

    #Clean location
    df["location"] = df["location"].astype(str).str.strip().str.lower()

    #Convert numeric
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
    df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    #Drop missing values
    df = df.dropna()

    #Remove invalid values
    df = df[
        (df["sqft"] > 0) &
        (df["bath"] > 0) &
        (df["bhk"] > 0) &
        (df["price"] > 0)
    ]

    return df

if __name__ == "__main__":
    data = preprocess_rent_data("data/raw/data.csv", "Bangalore")

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    output_path = "data/processed/bangalore_rent_clean.csv"
    data.to_csv(output_path, index=False)

    print("✅ Saved cleaned rent data:", data.shape)