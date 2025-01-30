import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from src.evaluate import regression_performance
from src.evaluate import regression_evaluation_plots


def model_performance_body():
    st.write("#### ML Model Summary")

    # Define file paths
    version = "v1"
    base_path = (
        "jupyter_notebooks/outputs/pipelines/train-test"
    )
    feature_importance_path = (
        "jupyter_notebooks/outputs/pipelines/"
        "feature_importance.png"
    )
    predicted_vs_actual_plot_path = (
        "jupyter_notebooks/outputs/pipelines/"
        "predicted_vs_actual_prices.png"
    )
    model_path = (
        "jupyter_notebooks/outputs/best_model/"
        "optimized_random_forest_model.pkl"
    )

    # ML Pipeline Overview
    st.info(
        "#### ML Pipeline Requirements:\n"
        "- The client requested an *R2* score of at least 0.75 on both "
        "training and test sets.\n"
        "- The model is trained and optimized using the Random Forest "
        "Regressor.\n\n"
        "**Evaluation Results:**\n\n"
        "- The training set achieved an R2 score of 0.961, which exceeds "
        "the required threshold.\n"
        "- The test set achieved an R2 score of 0.866, confirming the model's "
        "generalization is well within the agreed limits."
    )

    st.write("#### ML Pipeline Steps")
    with st.expander("View Pipeline Details"):
        st.code(
            """
            Pipeline(steps=[
                ('feature_transformation',
                 ColumnTransformer(transformers=[
                     ('yeo_johnson',
                      YeoJohnsonTransformer(variables=['GarageArea'])),
                     ('scaler', StandardScaler(),
                      ['GrLivArea', 'TotalBsmtSF']),
                     ('passthrough', 'passthrough',
                      ['OverallQual', 'YearRemodAdd'])
                 ], verbose_feature_names_out=False))
            ])
            """,
            language="python",
        )

    # Load the optimized model
    try:
        optimized_model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading the optimized model: {e}")
        return

    # Feature Importance
    st.write("---")
    st.write("#### Feature Importance")
    st.success("The best features")
    try:
        feature_importance = plt.imread(feature_importance_path)
        st.image(
            feature_importance,
            caption="Feature Importance",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Error loading feature importance image: {e}")

    # Model Evaluation Metrics
    st.write("---")
    st.subheader("Model Evaluation Metrics")
    with st.expander("View Performance Metrics"):
        regression_performance(model=optimized_model)

    # Predicted vs Actual Scatterplots
    with st.expander("View Predicted vs Actual Scatterplots"):
        st.success(
            """
            **Comment on price range performance:**
            The model performs very well for house prices in the lower and
            mid-range price segments, as indicated by the data points
            closely aligned with the red line.
            However, for higher price ranges, the model seems to slightly
            underestimate prices, as seen in the test data where data
            points tend to fall below the red line.
            """
        )
        regression_evaluation_plots(model=optimized_model, alpha_scatter=0.5)


if __name__ == "__main__":
    model_performance_body()
