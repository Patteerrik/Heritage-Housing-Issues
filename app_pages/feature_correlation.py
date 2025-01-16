import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feature_correlation_body():
    # Load CSV-file
    df = pd.read_csv('jupyter_notebooks/outputs/datasets/collection/HousePricesPredictionFeatures_Cleaned.csv')

    # Set the title and description for the page
    st.title("Feature Correlation with Sale Price")
    st.write("This page presents the most important features correlated with the sale price of the houses.")

    # Calculate the correlation with SalePrice
    correlation = df.corr()
    correlation_with_saleprice = correlation['SalePrice'].sort_values(ascending=False)

    # Get the top 5 correlated features with SalePrice
    top_6_features = correlation_with_saleprice.head(6)
    
    st.write("#### Correlation with SalePrice for Top 5 Features")

    # Create a new DataFrame for the heatmap
    top_6_df = top_6_features.to_frame()  # Convert the series to a DataFrame

    # Create and display a heatmap showing only the correlation of each feature with SalePrice
    fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size for better clarity
    sns.heatmap(top_6_df.T, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, cbar=True, linewidths=0.5)
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

        st.info(
            "The analysis successfully addressed the client's requirement to understand how different house attributes correlate with sale prices. "
            "We conducted a detailed correlation analysis to identify the most relevant features, visualized these correlations, and summarized the key insights."
        )

        st.success(
            "The analysis highlights that the features most strongly correlated with sale price include **OverallQual**, **GrLivArea**, **TotalBsmtSF**, **GarageArea**, and **YearRemodAdd**. "
            "Here’s how they impact the sale prices:"
            "\n- **OverallQual (0.79 correlation)**: This suggests a strong positive correlation, indicating that higher overall quality significantly increases the sale price."
            "\n- **GrLivArea (0.71 correlation)** and **TotalBsmtSF (0.61 correlation)**: Larger living areas and basement sizes are also positively correlated with higher sale prices, confirming that buyers value more spacious homes."
            "\n- **GarageArea (0.61 correlation)**: Similarly, a larger garage space contributes positively to the home's value."
            "\n- **YearRemodAdd (0.51 correlation)**: Recent remodels add to the sale price, suggesting that newer features or updates are important to buyers."
            "\nThese insights confirm that both the size and quality of various home features are critical in influencing house prices in Ames, Iowa."
        )



    # Methodology
    st.write("#### Methodology")
    st.info(
        "The correlation was calculated using Pearson's correlation coefficient, which shows how strongly the features are related to the sale price."
        " A positive correlation means that as the feature value increases, the sale price also tends to increase."
    )
