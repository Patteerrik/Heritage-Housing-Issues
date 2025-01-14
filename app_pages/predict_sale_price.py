import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict_sale_price_body():
    # Load the trained model
    model = joblib.load('jupyter_notebooks/outputs/best_model/best_gradient_boosting_model.pkl')

    # Load the saved pipeline
    pipeline = joblib.load("jupyter_notebooks/outputs/pipelines/feature_pipeline.pkl")

    # Define the feature order used by the model
    model_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']

    # Section 1: Display inherited houses data
    st.write("### Inherited Properties")
    df_inherited = pd.read_csv("inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/inherited_houses.csv")
    st.write("**Inherited house properties:**")
    st.dataframe(df_inherited)

    try:
        # Filter only the required features
        df_inherited_filtered = df_inherited[model_features]

        # Ensure positive values for Box-Cox transformation
        df_inherited_filtered['GrLivArea'] = df_inherited_filtered['GrLivArea'].apply(lambda x: x + 1 if x <= 0 else x)
        df_inherited_filtered['TotalBsmtSF'] = df_inherited_filtered['TotalBsmtSF'].apply(lambda x: x + 1 if x <= 0 else x)

        # Separate features to transform and untouched ones
        features_to_transform = ['GrLivArea', 'GarageArea', 'TotalBsmtSF']
        untouched_features = ['OverallQual']

        # Transform only the necessary features
        transformed_data = pd.DataFrame(
            pipeline.transform(df_inherited_filtered[features_to_transform]),
            columns=features_to_transform
        )

        # Combine transformed features with untouched ones
        input_data_transformed = pd.concat([df_inherited_filtered[untouched_features].reset_index(drop=True), transformed_data], axis=1)

        # Display transformed data
        st.write("Transformed data (inherited houses):")
        st.dataframe(input_data_transformed.head())

        # Predict prices
        predictions = model.predict(input_data_transformed)
        df_inherited['Predicted Sale Price'] = predictions
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

    if st.button("Predict Sale Price"):
        try:
            # Separate features to transform and untouched ones
            transformed_input = pd.DataFrame(
                pipeline.transform(input_data_df[features_to_transform]),
                columns=features_to_transform
            )

            # Combine transformed features with untouched ones
            final_input_data = pd.concat([input_data_df[untouched_features].reset_index(drop=True), transformed_input], axis=1)

            # Predict sale price
            prediction = model.predict(final_input_data)
            predicted_price = prediction[0]
            st.write(f"The predicted sale price is: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

