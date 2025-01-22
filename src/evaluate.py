import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def regression_performance(model):
    """Evaluate the model's performance using predefined train-test data."""
    # Load train-test data
    X_train = pd.read_csv("jupyter_notebooks/outputs/pipelines/train-test/v1/X_train.csv")
    X_test = pd.read_csv("jupyter_notebooks/outputs/pipelines/train-test/v1/X_test.csv")
    y_train = pd.read_csv("jupyter_notebooks/outputs/pipelines/train-test/v1/y_train.csv").squeeze()
    y_test = pd.read_csv("jupyter_notebooks/outputs/pipelines/train-test/v1/y_test.csv").squeeze()

    st.write("#### Model Evaluation \n")
    st.write("**Train Set**")
    regression_evaluation(X_train, y_train, model)
    st.write("**Test Set**")
    regression_evaluation(X_test, y_test, model)

def regression_evaluation(X, y, model):
    """Compute and display performance metrics for a single dataset."""
    prediction = model.predict(X)
    st.write('R2 Score:', r2_score(y, prediction).round(3))
    st.write('Mean Absolute Error:', mean_absolute_error(y, prediction).round(3))
    st.write('Mean Squared Error:', mean_squared_error(y, prediction).round(3))
    st.write('Root Mean Squared Error:', np.sqrt(mean_squared_error(y, prediction)).round(3))


def regression_evaluation_plots(model, alpha_scatter=0.5):
    """
    Display scatterplots for actual vs predicted values for training and test sets.
    Data is loaded internally.
    """
    # Load train-test data
    X_train = pd.read_csv("jupyter_notebooks/outputs/pipelines/train-test/v1/X_train.csv")
    X_test = pd.read_csv("jupyter_notebooks/outputs/pipelines/train-test/v1/X_test.csv")
    y_train = pd.read_csv("jupyter_notebooks/outputs/pipelines/train-test/v1/y_train.csv").squeeze()
    y_test = pd.read_csv("jupyter_notebooks/outputs/pipelines/train-test/v1/y_test.csv").squeeze()

    # Generate predictions
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Create scatterplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("Train Set")

    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Test Set")

    st.pyplot(fig)
