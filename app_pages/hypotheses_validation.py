import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def hypotheses_validation_body():
    st.title("Hypotheses Validation")
    st.write("This section presents the hypotheses tested in this project"
             " and how they were validated.")

    # Load dataset
    df = pd.read_csv('jupyter_notebooks/outputs/datasets/collection/'
                     'HousePricesFeatures.csv')

    # Hypothesis 1
    if st.button("Show Hypothesis 1: Larger houses have higher sale prices"):
        st.write("### Hypothesis 1: Larger houses have higher sale prices.")
        st.write("**Assumption**: Bigger houses, measured by the above-ground"
                 " living area (`GrLivArea`), tend to have higher sale prices"
                 " compared to smaller houses.")
        st.write("**Validation**: A correlation study was conducted between"
                 " the above-ground living area and the sale price, with"
                 " the relationship visualized using scatter plots.")

        # Show plot for Hypothesis 1
        st.write("#### Above-Ground Living Area (`GrLivArea`) vs Sale Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], ax=ax)
        sns.regplot(x=df['GrLivArea'], y=df['SalePrice'], ax=ax,
                    scatter=False, line_kws={'color': 'red', 'linewidth': 2})
        ax.set_title('Above-Ground Living Area (GrLivArea) vs Sale Price')
        ax.set_xlabel('Above-Ground Living Area (GrLivArea)')
        ax.set_ylabel('Sale Price')
        st.pyplot(fig)

    # Hypothesis 2
    if st.button("Show Hypothesis 2: Higher quality homes have"
                 " higher sale prices"):
        st.write("### Hypothesis 2: Higher quality homes have higher"
                 " sale prices.")
        st.write("**Assumption**: Homes with higher quality ratings"
                 " (`OverallQual`) tend to sell for higher prices compared"
                 " to homes with lower quality ratings.")
        st.write("**Validation**: A correlation analysis was conducted"
                 " and scatter plots were created to visualize the"
                 " relationship between `OverallQual` and `SalePrice`.")

        # Show plot for Hypothesis 2
        st.write("#### Overall Quality (`OverallQual`) vs Sale Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df['OverallQual'], y=df['SalePrice'], ax=ax)
        sns.regplot(x=df['OverallQual'], y=df['SalePrice'], ax=ax,
                    scatter=False, line_kws={'color': 'red', 'linewidth': 2})
        ax.set_title('Overall Quality (OverallQual) vs Sale Price')
        ax.set_xlabel('Overall Quality (OverallQual)')
        ax.set_ylabel('Sale Price')
        st.pyplot(fig)

    # Hypothesis 3
    if st.button("Show Hypothesis 3: Larger garages lead to"
                 " higher sale prices"):
        st.write("### Hypothesis 3: Larger garages lead to higher"
                 " sale prices.")
        st.write("**Assumption**: Homes with larger garage areas"
                 " (`GarageArea`) will sell for higher prices because"
                 " buyers value bigger garages.")
        st.write("**Validation**: A correlation analysis was performed"
                 " and scatter plots were created to explore how"
                 " `GarageArea` impacts the sale price.")

        # Show plot for Hypothesis 3
        st.write("#### Garage Area (`GarageArea`) vs Sale Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df['GarageArea'], y=df['SalePrice'], ax=ax)
        sns.regplot(x=df['GarageArea'], y=df['SalePrice'], ax=ax,
                    scatter=False, line_kws={'color': 'red', 'linewidth': 2})
        ax.set_title('Garage Area (GarageArea) vs Sale Price')
        ax.set_xlabel('Garage Area (GarageArea)')
        ax.set_ylabel('Sale Price')
        st.pyplot(fig)

    # Final thoughts or summary
    st.write("### Summary of Validation Results")
    st.write("- **Larger houses (`GrLivArea`)**: Bigger houses tend"
             " to sell for higher prices due to their increased living"
             " space.")
    st.write("- **Higher quality homes (`OverallQual`)**: Homes with"
             " better quality ratings achieve higher sale prices.")
    st.write("- **Larger garages (`GarageArea`)**: Buyers value"
             " bigger garages, leading to higher sale prices for"
             " such homes.")
    st.success("These findings provide clear insights into the factors"
               " that affect property prices in Ames, Iowa, and support"
               " data-driven decision-making.")
