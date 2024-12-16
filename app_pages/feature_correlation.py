import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feature_correlation_body():
    # Ladda data från en CSV-fil (ändra sökvägen till din fil)
    df = pd.read_csv('jupyter_notebooks/outputs/datasets/collection/HousePricesFeatures.csv')  # Ersätt med din egna datakälla

    # Set the title and description for the page
    st.title("Feature Correlation with Sale Price")
    st.write("This page presents the most important features correlated with the sale price of the houses.")

    # Calculate the correlation with SalePrice
    correlation = df.corr()
    correlation_with_saleprice = correlation['SalePrice'].sort_values(ascending=False)

    # Display the top correlated features
    st.write("#### Top Correlated Features")
    st.dataframe(correlation_with_saleprice.head(10))  # Show the top 10 correlated features

    # Create a button to show/hide the plots
    show_plots = st.button("Show Correlation Plots")

    if show_plots:
        # Create visualizations for each top feature in separate plots
        st.write("#### Visualizations")
        top_features = correlation_with_saleprice.head(5).index  # Get the top 5 correlated features

        # Create a separate plot for each top feature
        for feature in top_features:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create scatter plot for each feature
            sns.scatterplot(x=df[feature], y=df['SalePrice'], ax=ax, label=feature, s=80)

            # Add a regression line for each feature
            sns.regplot(x=df[feature], y=df['SalePrice'], ax=ax, scatter=False, line_kws={'linewidth': 2, 'color': 'red'})

            # Set the title, axis labels, and legend
            ax.set_title(f"{feature} vs Sale Price", fontsize=16)
            ax.set_xlabel(f"{feature} Values", fontsize=12)
            ax.set_ylabel("Sale Price", fontsize=12)
            ax.legend(title="Features")

            # Display each plot separately in Streamlit
            st.pyplot(fig)

    # Interpretation and conclusions
    st.write("#### Interpretation and Conclusions")
    st.success(
        f"Based on our analysis, the most strongly correlated features with sale price are GrLivArea and OverallQual."
        f" These features have a high positive correlation, indicating that larger living area and better overall quality"
        f" lead to higher sale prices. Other features like GarageArea and BsmtFinSF1 also show moderate positive correlation."
        f" Additionally, TotalSquareFootage shows a strong positive correlation with sale price, indicating that larger total square footage"
        f" tends to result in higher sale prices as well."
    )

    # Methodology
    st.write("#### Methodology")
    st.info(
        "The correlation was computed using Pearson's correlation coefficient, which measures the linear relationship "
        "between the features and the sale price. A positive correlation indicates that as the feature increases, so does the sale price."
    )