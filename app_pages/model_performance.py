import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def model_performance_body():
    st.write("#### ML Model Summary")

    # Define file paths
    version = "v1"
    base_path = "jupyter_notebooks/outputs/pipelines/train-test"
    feature_importance_path = "jupyter_notebooks/outputs/pipelines/feature_importance.png"
    predicted_vs_actual_plot_path = "jupyter_notebooks/outputs/pipelines/predicted_vs_actual_prices.png"

    X_train_path = f"{base_path}/{version}/X_train.csv"
    X_test_path = f"{base_path}/{version}/X_test.csv"
    y_train_path = f"{base_path}/{version}/y_train.csv"
    y_test_path = f"{base_path}/{version}/y_test.csv"

    # Load data
    try:
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)
    except Exception as e:
        st.error(f"Error loading train-test datasets: {e}")
        return

    # ML Pipeline Overview
    st.info(
        f"#### ML Pipeline Requirements:\n"
        f"* The client requested an *R2* score of at least 0.75 on both training and test sets.\n"
        f"* **Train set:** The training data includes {len(X_train)} samples.\n"
        f"* **Test set:** The test data includes {len(X_test)} samples.\n\n"
        f"**Feature importance analysis is provided below.**"
    )
    
    st.write("#### ML Pipeline Steps")
    with st.expander("View Pipeline Details"):
        st.code("""
        Pipeline(steps=[
            ('feature_transformation',
             ColumnTransformer(transformers=[
                 ('yeo_johnson', YeoJohnsonTransformer(variables=['GarageArea'])),
                 ('scaler', StandardScaler(), ['GrLivArea', 'TotalBsmtSF']),
                 ('passthrough', 'passthrough', ['OverallQual', 'YearRemodAdd'])
             ], verbose_feature_names_out=False))
        ])
        """, language="python")


    # Feature Importance
    st.write("---")
    st.write("#### Feature Importance")
    try:
        feature_importance = plt.imread(feature_importance_path)
        st.image(feature_importance, caption="Feature Importance", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading feature importance image: {e}")

    # Predicted vs Actual Prices Scatterplot
    st.write("---")
    st.write("#### Predicted vs Actual Prices (Training and Test Sets)")
    try:
        st.image(predicted_vs_actual_plot_path, 
                 caption="Predicted vs Actual Prices (Training and Test Sets)", 
                 use_column_width=True)
    except Exception as e:
        st.error(f"Error loading predicted vs actual prices plot: {e}")
