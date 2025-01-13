import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict_sale_price_body():
    # Load the trained model
    model = joblib.load('jupyter_notebooks/outputs/best_model/best_xgboost_model.pkl')

    # Load the saved pipeline
    pipeline = joblib.load("jupyter_notebooks/outputs/pipelines/feature_pipeline.pkl")

    # Define the feature order used by the model
    model_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']

    # Section 1: Display inherited houses data
    st.write("### Inherited Properties")
    df_inherited = pd.read_csv("inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/inherited_houses.csv")
    st.write("**Inherited house properties:**")
    st.dataframe(df_inherited)

    # Filter data based on the model's features
    df_inherited_filtered = df_inherited[model_features]

    # Transform the data and predict prices
    try:
        input_data_transformed = pipeline.transform(df_inherited_filtered)
        print("Transformed data (inherited houses):\n", input_data_transformed[:5])  # Lägg till denna rad
        predictions = model.predict(input_data_transformed)
        df_inherited['Predicted Sale Price'] = np.expm1(predictions)  # Om log-transform användes
    except Exception as e:
       st.error(f"Error during prediction: {e}")
       return

    # Section 2: Display predicted prices
    st.write("### Predicted Inherited Property Prices")
    st.write("**Predicted prices for inherited houses:**")
    st.dataframe(df_inherited[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'Predicted Sale Price']])

    # Display the total predicted price
    total_price = df_inherited['Predicted Sale Price'].sum()
    st.write(f"**Total predicted sale price for all inherited houses: ${total_price:,.2f}**")

    # Section 3: Manual prediction for a new house
    st.write("---")
    st.write("### Predict Sale Price for a New House")
    st.write("Enter the details of the house to predict its sale price.")

    # Create input fields for manual house feature inputs
    input_data = {}
    for feature in model_features:
        if feature == 'OverallQual':  # Special handling for categorical-like sliders
            input_data[feature] = st.slider(f"{feature} (1 to 10)", 1, 10, 5)
        else:
            input_data[feature] = st.number_input(f"{feature} (Enter value)", min_value=1)

    # Convert input data to DataFrame in correct feature order
    input_data_df = pd.DataFrame([input_data], columns=model_features)

    # Convert input data to DataFrame in correct feature order
    input_data_df = pd.DataFrame([input_data], columns=model_features)

    # Display a button to make the prediction
    if st.button("Predict Sale Price"):
        try:
            # Transform the input data
            input_data_transformed = pipeline.transform(input_data_df)
            # Make the prediction
            prediction = model.predict(input_data_transformed)
            # Convert prediction to original scale (inverse log transform if applicable)
            predicted_price = np.expm1(prediction[0])  # Remove np.expm1 if no log-transform was applied
            st.write(f"The predicted sale price is: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

