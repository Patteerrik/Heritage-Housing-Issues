import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict_sale_price_body():
    # Load the trained model 
    model = joblib.load('jupyter_notebooks/outputs/best_model/best_xgboost_model.pkl')

    # Set the title and description for the page
    st.title("Predict Sale Price of a House")
    st.write("Enter the details of the house to predict its sale price.")

    # Create input fields for house features
    gr_liv_area = st.number_input("GrLivArea (Above ground living area in square feet)", min_value=0)
    overall_quality = st.slider("Overall Quality (1 to 10)", 1, 10, 5)
    garage_area = st.number_input("GarageArea (Area of the garage in square feet)", min_value=0)
    bsmt_fin_sf1 = st.number_input("BsmtFinSF1 (Finished area of the basement in square feet)", min_value=0)

    # Prepare the input data as a list and ensure it matches the model's expected input
    input_data = np.array([gr_liv_area, overall_quality, garage_area, bsmt_fin_sf1]).reshape(1, -1)

    # Display a button to make the prediction
    if st.button("Predict Sale Price"):
        # Make the prediction
        prediction = model.predict(input_data)
        # Convert prediction to original scale (inverse log transform)
        predicted_price = np.expm1(prediction[0])
        st.write(f"The predicted sale price is: ${predicted_price:,.2f}")

    # Add some explanation
    st.write("### How it works:")
    st.info("This page allows you to predict the sale price of a house based on the most impactful features: living area, overall quality, garage size, and finished basement area. The model uses these features to estimate the price based on the house price data it was trained on.")

    # Show a sample dataset or description (optional)
    st.write("### Sample Data")
    st.write("Here is an example of how the input features relate to the model prediction.")

    # Example sample data
    sample_data = {
        "GrLivArea": [1500, 2500, 1800],
        "Overall Quality": [5, 8, 7],
        "GarageArea": [500, 600, 450],
        "BsmtFinSF1": [800, 1200, 1000]
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)
