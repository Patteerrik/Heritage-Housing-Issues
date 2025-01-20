import streamlit as st
import pandas as pd
import joblib

def predict_sale_price_body():
    # Load the trained model and pipeline
    model_path = 'jupyter_notebooks/outputs/best_model/optimized_random_forest_model.pkl'
    pipeline_path = 'jupyter_notebooks/outputs/pipelines/feature_pipeline_cleaned.pkl'
    
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    # Define the features used in the model
    best_features = ['GarageArea', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearRemodAdd']

    # Title and Business Requirement 2
    st.info("""
    ### Predict House Sale Prices  
    This page addresses **Business Requirement 2**, which involves creating a machine learning model to:  
    - Predict the sale price of the client's four inherited houses and other houses in Ames, Iowa.  
    - Train and optimize a predictive model to ensure accuracy and reliability.  

    The predictions on this page support data driven decision making for the client.
    """)

    # --- Section 1: Prediction for Inherited Houses ---
    st.write("#### Inherited Properties")

    # Load data for inherited houses
    df_inherited = pd.read_csv("jupyter_notebooks/outputs/datasets/collection/InheritedHouses.csv")
    st.write("**Inherited house properties:**")
    st.dataframe(df_inherited)

    try:
        # Filter and transform data for inherited houses
        df_filtered = df_inherited[best_features]
        transformed_data = pd.DataFrame(
            pipeline.transform(df_filtered),
            columns=best_features
        )

        # Predict prices
        predictions = model.predict(transformed_data)
        df_inherited['Predicted Sale Price'] = predictions

        # Display results
        st.write("### Predicted Prices for Inherited Properties")
        st.dataframe(df_inherited[['GarageArea', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearRemodAdd', 'Predicted Sale Price']])
        
        # Summarize predicted prices
        total_price = df_inherited['Predicted Sale Price'].sum()
        st.write(f"**Total predicted sale price for all inherited houses: ${total_price:,.2f}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # --- Section 2: Prediction for New Data ---
    st.write("---")
    st.write("### Predict Sale Price for a New Property")
    st.success("""  
    This feature fulfills **Business Requirement 2**, which focuses on creating a machine learning model to predict house prices for inherited houses and other properties in Ames, Iowa.  
    The prediction is made using the **best features** identified during model training to ensure accuracy and reliability.
    """)

    input_data = {}
    for feature in best_features:
        if feature == 'OverallQual':
            input_data[feature] = st.slider(f"{feature} (1 to 10)", 1, 10, 5)
        elif feature == 'YearRemodAdd':
            input_data[feature] = st.number_input(f"{feature} (Year)", min_value=1900, max_value=2025, value=2000)
        else:
            input_data[feature] = st.number_input(f"{feature} (Enter value)", min_value=1, value=100)

    # Convert input to DataFrame
    input_data_df = pd.DataFrame([input_data], columns=best_features)

    # Predict price for new data
    if st.button("Predict Sale Price"):
        try:
            transformed_input = pd.DataFrame(
                pipeline.transform(input_data_df),
                columns=best_features
            )
            prediction = model.predict(transformed_input)
            predicted_price = prediction[0]
            st.write(f"**The predicted sale price for the property is: ${predicted_price:,.2f}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

