import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict_sale_price_body():
    # Load the trained model
    try:
        model = joblib.load('jupyter_notebooks/outputs/best_model/optimized_gradient_boosting_model.pkl')
        st.write("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return

    # Load the saved pipeline
    try:
        pipeline = joblib.load("jupyter_notebooks/outputs/pipelines/feature_pipeline.pkl")
        st.write("✅ Pipeline loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {e}")
        return

    # Define the feature order used by the model
    model_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearRemodAdd']
    st.write(f"Expected features by the model: {model_features}")

    # Section 1: Display inherited houses data
    st.write("### Inherited Properties")
    try:
        df_inherited = pd.read_csv("inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/inherited_houses.csv")
        st.write("**Inherited house properties:**")
        st.dataframe(df_inherited)
        st.write(f"Loaded inherited data shape: {df_inherited.shape}")
    except Exception as e:
        st.error(f"❌ Error loading inherited data: {e}")
        return

    # Check if required features are present
    missing_features = [feature for feature in model_features if feature not in df_inherited.columns]
    if missing_features:
        st.error(f"❌ Missing columns in inherited data: {missing_features}")
        return
    else:
        st.write("✅ All required features are present in the dataset.")

    try:
        # Filter only the required features
        df_inherited_filtered = df_inherited[model_features]
        st.write("Filtered inherited data (before transformation):")
        st.dataframe(df_inherited_filtered)
    except Exception as e:
        st.error(f"❌ Error filtering data: {e}")
        return

    try:
        # Transform data using the pipeline
        transformed_data = pd.DataFrame(
            pipeline.transform(df_inherited_filtered),
            columns=pipeline.named_steps['feature_transformation'].get_feature_names_out()
        )
        st.write("Transformed data (inherited houses):")
        st.dataframe(transformed_data.head())
    except Exception as e:
        st.error(f"❌ Error during transformation: {e}")
        return

    try:
        # Predict prices
        predictions = model.predict(transformed_data)
        df_inherited['Predicted Sale Price'] = predictions
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return

    # Section 2: Display predicted prices
    st.write("### Predicted Inherited Property Prices")
    st.write("**Predicted prices for inherited houses:**")
    st.dataframe(df_inherited[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearRemodAdd', 'Predicted Sale Price']])

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
            # Transform input data using the pipeline
            transformed_input = pd.DataFrame(
                pipeline.transform(input_data_df),
                columns=pipeline.named_steps['feature_transformation'].get_feature_names_out()
            )

            # Predict sale price
            prediction = model.predict(transformed_input)
            predicted_price = prediction[0]
            st.write(f"The predicted sale price is: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"❌ Error during manual prediction: {e}")

# Call the prediction function
predict_sale_price_body()
