#Import libraries
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
#Import from sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#Load cleaned data
def load_data():
    return pd.read_csv("data/processed/bangalore_rent_clean.csv")

#Train models
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100,
                                               max_depth=10,
                                               min_samples_split=5,
                                               min_samples_leaf=2,
                                               random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100,
                                                       learning_rate=0.05,
                                                       max_depth=3,
                                                       random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

#Evaluate models
def evaluate_models(models,X_train, X_test, y_train, y_test):
    results = {}

    print("\n========== MODEL PERFORMANCE ==========\n")

    for name, model in models.items():
        #Prediction
        preds = model.predict(X_test)

        #Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        #Cross-validation (R2) k=5
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_r2 = cv_scores.mean()

        #Store
        results[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_r2": cv_r2
        }

        #Clean output
        print(f"{name}")
        print(f"    MAE     : {mae:.2f}")
        print(f"    RMSE    : {rmse:.2f}")
        print(f"    R2      : {r2:.4f}")
        print(f"    CV_R2   : {cv_r2:.4f}")
        print("-" * 40)

    return results

#Select best model (based on RMSE)
def select_best_model(models, results):
    best_model_name = max(results, key=lambda x: results[x]["cv_r2"])

    print(f"\nBest Model: {best_model_name}")

    return models[best_model_name], best_model_name

#Save model
def save_model(model):
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/rent_model.pkl")
    print("Rent model saved at models/rent_model.pkl")

#Main Pipeline
if __name__ == "__main__":
    df = load_data()

    X = df[["location", "sqft", "bath", "bhk"]]
    y = df["price"]

    #One-hot encoding
    X = pd.get_dummies(X, columns=["location"])

    #Save feature columns
    joblib.dump(X.columns, "models/rent_feature_columns.pkl")

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Train
    models = train_models(X_train, y_train)

    #Evaluate
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    #Select best
    best_model, best_name = select_best_model(models, results)

    #Save
    save_model(best_model)