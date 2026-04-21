YAML
---
title: House Price Predictor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

🏠 House Price Predictor

A production-style Machine Learning project that predicts house prices based on location, square footage, bedrooms, and bathrooms.

This project demonstrates an end-to-end ML pipeline, from data preprocessing to model deployment with a Streamlit UI.

---

🚀 Features

- 📊 Data preprocessing and cleaning pipeline
- 🧠 Multiple model training (Linear Regression, Random Forest, Gradient Boosting)
- 📈 Model selection using Cross Validation
- 🧩 Feature engineering (location-based pricing)
- 🏙️ Location-aware predictions using aggregated features
- 🖥️ Interactive Streamlit web app
- 📦 Model serialization and inference pipeline

---

🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Streamlit

---

📂 Project Structure

house-price-predictor/
│
├── app.py                     # Streamlit UI
├── requirements.txt          # Dependencies
├── Dockerfile                # Deployment config
├── README.md                 # Project documentation
│
├── src/
│   ├── preprocess.py         # Data preprocessing pipeline
│   ├── train.py              # Model training & evaluation
│   ├── predict.py            # Inference pipeline
│
├── models/
│   ├── best_model.pkl
│   ├── feature_columns.pkl
│   ├── location_avg_price.pkl
│
├── data/
│   └── raw/                  # Dataset

---

⚙️ How It Works

1. Data Preprocessing

- Handles missing values
- Converts "total_sqft" into numeric format
- Extracts bedrooms from "size"
- Removes outliers
- Creates "price_per_sqft"
- Generates location-based aggregated features

---

2. Feature Engineering

Key feature:

location_avg_price

This feature captures average price per sqft per location, allowing the model to properly understand location impact.

---

3. Model Training

Models used:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Model selection is based on:

Cross-Validation R² Score (CV_R2)

---

4. Inference Pipeline

During prediction:

- Input is transformed using saved feature schema
- Location-based feature ("location_avg_price") is injected
- Model predicts price in Lakhs

---

🚀 Live Demo

Coming soon (Hugging Face Deployment in progress)

---

📊 Example Prediction

Input:
Location: Whitefield
Sqft: 1200
Bedrooms: 2
Bathrooms: 2

Output:
₹70.04 Lakhs

---

🚀 Deployment

This project is deployed using:

- Streamlit
- Hugging Face Spaces (Docker-based)

---

📌 Key Learning Outcomes

- Handling real-world messy datasets
- Feature engineering for tabular ML
- Model evaluation using cross-validation
- Aligning training and inference pipelines
- Debugging feature mismatch issues
- Building deployable ML applications

---

🤝 Future Improvements

- Add multiple cities support
- Improve model accuracy with advanced features
- Add explainability (SHAP / feature importance UI)
- Convert into full house-hunting assistant (agent-based system)

---

👨‍💻 Author

Aadhar Kaushik

- AI/ML Engineer
- Python & Machine Learning Mentor
- Focused on building real-world AI/ML systems

---

⭐ If you like this project

Give it a star ⭐ and feel free to fork it!
