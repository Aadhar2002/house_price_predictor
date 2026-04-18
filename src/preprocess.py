#Import libraries
import pandas as pd
import numpy as np

#Load data
def load_data(file_path: str)-> pd.DataFrame:
    """
    Load raw housing data from CSV.
    """
    return pd.read_csv(file_path)

#Clean the data
def clean_basic_column(df: pd.DataFrame)-> pd.DataFrame:
    """
    Keep only the columns needed for the MVP.
    """
    required_columns = ["location", "size", "total_sqft", "bath", "price"]
    df = df[required_columns].copy()
    return df

#Handle missing value
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values in important columns.
    """
    df = df.dropna()
    return df

#Extreact bedrooms
def extract_bedroom(df: pd.DataFrame)-> pd.DataFrame:
    """
    Extract bedroom number from the 'size' column.
    Example:
        '2 BHK' -> 2
        '4 Bedroom' -> 4
    """
    df["bedroom"] = df["size"].str.strip().str.split().str[0].astype(int)
    return df

#Convert sqft to number
def convert_sqft_to_num(value):
    """
    Convert total_sqft values into numeric form.
    Handles:
        - '1200'
        - '1133-1384' -> average
        - invalid values -> Nan
    """
    try:
        if isinstance(value, str):
            value = value.strip()
            if "-" in value:
                parts = value.split("-")
                if len(parts) == 2:
                    return (float(parts[0]) + float(parts[1]))/2
                return float(value)
            return float(value)
    except:
        return np.nan

#clean total_sqft
def clean_total_sqft(df: pd.DataFrame)-> pd.DataFrame:
    """
    Create a numeric total_sqft column.
    """
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)
    df = df.dropna(subset=["total_sqft"])
    return df

#Clean location
def clean_location(df: pd.DataFrame)-> pd.DataFrame:
    """
    Strip spcaes from location names.
    """
    df["location"] = df["location"].apply(lambda x: x.strip())
    return df

#Reduce location categories
def reduce_location_categories(df: pd.DataFrame, min_count: int=25)-> pd.DataFrame:
    """
    Group rare locations into 'other'.
    """
    location_counts = df["location"].value_counts()
    rare_locations = location_counts[location_counts <=min_count].index
    df["location"] = df["location"].apply(lambda x: "other" if x in rare_locations else x)
    return df

#Remove sqft outliers
def remove_sqft_outliers(df: pd.DataFrame)-> pd.DataFrame:
    """
    Remove rows where square feet per bedroom is unrealistically low.
    Rule used:
        total_sqft/bedroom >= 300
    """
    df = df[df["total_sqft"] / df["bedroom"] >= 300]
    return df

#Remove bath outliers
def remove_bath_outliers(df: pd.DataFrame)-> pd.DataFrame:
    """
    Remove rows where number of bathrooms are unrealistically high.
    Example rule:
        bathrooms should not exceed bedroom + 2
    """
    df = df[df["bath"] <= df["bedroom"] + 2]
    return df

#Add price per sqft
def add_price_per_sqft(df:pd.DataFrame)-> pd.DataFrame:
    """
    Create price per square foot helper column.
    """
    df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]
    return df

#Remove price per sqft outliers
def remove_price_per_sqft_outliers(df:pd.DataFrame)-> pd.DataFrame:
    """
    Remove extreme price_per_sqft values.
    These thresholds are practical heuristics for this dataset.
    """
    df = df[(df["price_per_sqft"] >= 3000) & (df["price_per_sqft"] <= 30000)]
    return df

#Add derived features
def add_engineered_per_sqft_outliers(df: pd.DataFrame)-> pd.DataFrame:
    """
    Add useful engineered features.
    """
    df["sqft_per_bedroom"] = df["total_sqft"] / df["bedroom"]
    return df

#Final cleanup
def final_cleanup(df: pd.DataFrame)-> pd.DataFrame:
    """
    Drop original size column after extracting bedroom.
    """
    df = df.drop(columns=["size"])
    return df

#Preprocess data
def preprocess_data(file_path: str)-> pd.DataFrame:
    """
    Full preprocessing pipeline.
    """
    df = load_data(file_path)
    df = clean_basic_column(df)
    df = handle_missing_values(df)
    df = extract_bedroom(df)
    df = clean_total_sqft(df)
    df = clean_location(df)
    df = reduce_location_categories(df, min_count=25)
    df = remove_sqft_outliers(df)
    df = remove_bath_outliers(df)
    df = add_price_per_sqft(df)
    df = remove_price_per_sqft_outliers(df)
    df = add_engineered_per_sqft_outliers(df)
    df = final_cleanup(df)
    return df

if __name__ == "__main__":
    file_path = "data/raw/BHP.csv"
    processed_df = preprocess_data(file_path)

    print("Processed Data Shape:", processed_df.shape)
    print(processed_df.head())