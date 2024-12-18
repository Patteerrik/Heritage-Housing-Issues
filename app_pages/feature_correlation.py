import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feature_correlation_body():
    # Load CSV-file
    df = pd.read_csv('jupyter_notebooks/outputs/datasets/collection/HousePricesFeatures.csv')

    # Set the title and description for the page
    st.title("Feature Correlation with Sale Price")
    st.write("This page presents the most important features correlated with the sale price of the houses.")

    # Calculate the correlation with SalePrice
    correlation = df.corr()
    correlation_with_saleprice = correlation['SalePrice'].sort_values(ascending=False)

    # Get the top 10 correlated features with SalePrice
    top_10_features = correlation_with_saleprice.head(10)
    
    st.write("#### Correlation with SalePrice for Top 10 Features")

    # Create a new DataFrame for the heatmap (only the top 10 features and their correlation with SalePrice)
    top_10_df = top_10_features.to_frame()  # Convert the series to a DataFrame

    # Create and display a heatmap showing only the correlation of each feature with SalePrice
    fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size for better clarity
    sns.heatmap(top_10_df.T, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, cbar=True, linewidths=0.5)
    ax.set_title("Correlation with SalePrice", fontsize=16)
    st.pyplot(fig)

    # Create a button to show/hide the plots
    show_plots = st.checkbox("Show Correlation Plots", value=False)

    if show_plots:
        st.write("#### Visualizations")
        top_features = correlation_with_saleprice.head(6).index
        
        for feature in top_features:
            # Avoid plotting SalePrice vs SalePrice
            if feature != 'SalePrice':
                fig, ax = plt.subplots(figsize=(10, 6))

                # Scatter plot for each feature against SalePrice
                sns.scatterplot(x=df[feature], y=df['SalePrice'], ax=ax, label=feature, s=80)

                # Adding regression line
                sns.regplot(x=df[feature], y=df['SalePrice'], ax=ax, scatter=False, line_kws={'linewidth': 2, 'color': 'red'})

                ax.set_title(f"{feature} vs Sale Price", fontsize=16)
                ax.set_xlabel(f"{feature} Values", fontsize=12)
                ax.set_ylabel("Sale Price", fontsize=12)
                ax.legend(title="Features")

                st.pyplot(fig)


    # Interpretation and conclusions
    st.write("#### Interpretation and Conclusions")
    st.success(
        f"The analysis shows that the most strongly correlated features with sale price are **OverallQual** and **GrLivArea**."
        f" These features have a strong positive correlation, meaning that better quality and larger living area lead to higher sale prices."
        f" Other features like **GarageArea**, **BsmtFinSF1**, and **KitchenQual_Ex** also have a positive correlation with sale price."
        f" Larger garage areas and finished basements also help increase house prices. **GarageFinish_Fin** and **2ndFlrSF** also show a positive correlation,"
        f" but not as strong."
    )


    # Methodology
    st.write("#### Methodology")
    st.info(
        "The correlation was calculated using Pearson's correlation coefficient, which shows how strongly the features are related to the sale price."
        " A positive correlation means that as the feature value increases, the sale price also tends to increase."
    )
