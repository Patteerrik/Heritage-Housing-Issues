import streamlit as st
import pandas as pd
from joblib import load
from src.evaluate import regression_performance, regression_evaluation_plots

def model_performance_body():
    st.write("#### ML Model Summary")

    # Define file paths
    model_path = "jupyter_notebooks/outputs/best_model/optimized_random_forest_model.pkl"
    base_path = "jupyter_notebooks/outputs/pipelines/train-test/v1"
    feature_importance_path = "jupyter_notebooks/outputs/pipelines/feature_importance.png"

    X_train_path = f"{base_path}/X_train.csv"
    X_test_path = f"{base_path}/X_test.csv"
    y_train_path = f"{base_path}/y_train.csv"
    y_test_path = f"{base_path}/y_test.csv"

    # Load data
    try:
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).squeeze()  # Ensure y is 1D
        y_test = pd.read_csv(y_test_path).squeeze()   # Ensure y is 1D
        #st.success("Train and test datasets loaded successfully.")
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return

    # Load the optimized model
    try:
        optimized_model = load(model_path)
        st.success(f"Optimized model loaded successfully from {model_path}")
    except Exception as e:
        st.error(f"Error loading the optimized model: {e}")
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

    st.write("---")
    st.write("#### Feature Importance")
    st.image(feature_importance_path, caption="Feature Importance", use_container_width=True)

    # Expanders for results
    st.write("---")
    st.write("#### Model Evaluation")

    # Evaluate model performance
    with st.expander("View Model Performance Metrics"):
        regression_performance(X_train, y_train, X_test, y_test, model=optimized_model)

    # Display scatterplots
    with st.expander("View Predicted vs Actual Scatterplots"):
        regression_evaluation_plots(X_train, y_train, X_test, y_test, model=optimized_model, alpha_scatter=0.5)


if __name__ == "__main__":
    st.title("House Price Prediction Model Evaluation")
    model_performance_body()
