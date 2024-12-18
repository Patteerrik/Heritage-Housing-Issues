import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def hypotheses_validation_body():
    st.title("Hypotheses Validation")
    st.write("This section presents the hypotheses tested in this project and how they were validated.")

    # Hypothesis 1
    st.write("### Hypothesis 1: Larger houses have higher sale prices.")
    st.write("**Assumption**: Bigger houses, measured by the combined area of all floors, tend to have higher sale prices compared to smaller houses.")
    st.write("**Validation**: A correlation study was conducted between the total area of the house and its sale price, with the relationship visualized using scatter plots.")
    st.write("**Result**: The scatter plot and correlation analysis show a positive relationship between the total area of the house and its sale price, confirming the hypothesis that bigger houses tend to have higher sale prices.")
    
    # Show plot for Hypothesis 1 (Total Area vs SalePrice)
    df = pd.read_csv('jupyter_notebooks/outputs/datasets/collection/HousePricesFeatures.csv')
    st.write("#### Total Area vs Sale Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['GrLivArea'] + df['2ndFlrSF'], y=df['SalePrice'], ax=ax)

    # Add a red trendline
    sns.regplot(x=df['GrLivArea'] + df['2ndFlrSF'], y=df['SalePrice'], ax=ax, scatter=False, line_kws={'color': 'red', 'linewidth': 2})

    ax.set_title('Total Area vs Sale Price')
    ax.set_xlabel('Total Area (GrLivArea + 2ndFlrSF)')
    ax.set_ylabel('Sale Price')
    st.pyplot(fig)

    # Hypothesis 2
    st.write("### Hypothesis 2: Higher quality homes have higher sale prices.")
    st.write("**Assumption**: Homes with higher quality ratings ('OverallQual') tend to sell for higher prices compared to homes with lower quality ratings.")
    st.write("**Validation**: A correlation analysis was conducted and scatter plots were created to visualize the relationship between 'OverallQual' and 'SalePrice'.")
    st.write("**Result**: The scatter plot and correlation analysis show a strong positive link between OverallQual and SalePrice. This confirms that home quality affects its sale price.")
    
    # Show plot for Hypothesis 2 (OverallQual vs SalePrice)
    st.write("#### Overall Quality vs Sale Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['OverallQual'], y=df['SalePrice'], ax=ax)

    # Add a red trendline
    sns.regplot(x=df['OverallQual'], y=df['SalePrice'], ax=ax, scatter=False, line_kws={'color': 'red', 'linewidth': 2})

    ax.set_title('Overall Quality vs Sale Price')
    ax.set_xlabel('Overall Quality')
    ax.set_ylabel('Sale Price')
    st.pyplot(fig)

    # Hypothesis 3
    st.write("### Hypothesis 3: Larger garages lead to higher sale prices.")
    st.write("**Assumption**: Houses with larger garage areas ('GarageArea') will sell for higher prices because buyers value bigger garages.")
    st.write("**Validation**: A correlation analysis was performed and scatter plots were created to explore how garage area impacts the sale price.")
    st.write("**Result**: The scatter plot and correlation analysis show a positive relationship between 'GarageArea' and 'SalePrice'. This confirms that houses with bigger garages tend to sell for higher prices.")
    
    # Show plot for Hypothesis 3 (GarageArea vs SalePrice)
    st.write("#### Garage Area vs Sale Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['GarageArea'], y=df['SalePrice'], ax=ax)

    # Add a red trendline
    sns.regplot(x=df['GarageArea'], y=df['SalePrice'], ax=ax, scatter=False, line_kws={'color': 'red', 'linewidth': 2})
    
    ax.set_title('Garage Area vs Sale Price')
    ax.set_xlabel('Garage Area')
    ax.set_ylabel('Sale Price')
    st.pyplot(fig)

    # Final thoughts or summary
    st.write("### Summary of Validation Results")
    st.write("All three hypotheses were validated successfully with positive correlations between the variables and sale prices.")
    st.write("Larger houses, higher quality homes, and homes with larger garages all show clear evidence of higher sale prices.")

