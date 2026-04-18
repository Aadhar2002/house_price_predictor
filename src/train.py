#Import Libraries
import pandas as pd
import numpy as np
#Import from sklearn for pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from preprocess import preprocess_data
import joblib
import os

#Prepare features
def prepare_features(df):
    if "price_per_sqft" in df.columns:
        df = df.drop(columns=["price_per_sqft"])

    df = pd.get_dummies(df, columns=["location"], drop_first=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    return X,y

#Splitting data into train test data (80/20)
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        train_size = 0.8,
        random_state = 42
    )
    return X_train, X_test, y_train, y_test

#Model Training
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators = 100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

#Model Evaluation
def evaluate_models(models, X_train, X_test, y_train, y_test):

    results = {}

    for name, model in models.items():

        #Train model
        model.fit(X_train, y_train)

        #Test set prediction
        predictions = model.predict(X_test)

        #Mean Absolute Error
        mae = mean_absolute_error(y_test, predictions)

        #Mean squared error
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        #R2 Score
        r2 = r2_score(y_test, predictions)

        #Cross-Validation
        cross_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring = 'r2'
        )

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "CV_R2": cross_scores.mean()
        }

    return results

#Select best model
def select_best_model(models, results):
    """
    Select best model based on highest R2 score.
    """

    best_model_name = max(results, key=lambda x:results[x]["CV_R2"])
    best_model = models[best_model_name]

    return best_model_name, best_model

#Save model
def save_model(model, feature_columns):
    """
    Save trained model and feature columns.
    """
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/best_model.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    print("Model and feature columns saved successfully")

#Full training pipeline
def train_pipeline(file_path):
    print("Loading and preprocessing data.....")
    df = preprocess_data(file_path)

    print("Preparing features.....")
    X, y = prepare_features(df)

    print("Splitting dataset.....")
    X_train, X_test, y_train, y_test = split_data(X,y)

    print("Training model.....")
    models = train_models(X_train, y_train)

    print("Evaluating model.....")
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    best_model_name, best_model = select_best_model(models, results)

    print(f"\nBest Model: {best_model_name}")

    save_model(best_model, X.columns.tolist())
    return results, best_model_name


#Run training
if __name__ == "__main__":
    file_path = "data/raw/BHP.csv"

    results, best_model_name = train_pipeline(file_path)

    print("\nModel Performance\n")

    for model_name, metrics in results.items():
        print(model_name)
        print("MAE", metrics["MAE"])
        print("RMSE", metrics["RMSE"])
        print("R2", metrics["R2"])
        print("CV_R2:", metrics["CV_R2"])
        print("-------------------------")

    print(f"\nSaved Model: {best_model_name}")