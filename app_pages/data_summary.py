import streamlit as st

def data_summary_body():

    # Header for the project summary
    st.write("### Project Overview")

    # Summary of the dataset used in the project
    st.info(
        "**Dataset Overview**\n"
    "This project uses data from house sales in Ames, Iowa, containing information about "
    "various features that potentially impact the house prices. Some of the most strongly correlated features with "
    "sale prices include aspects such as overall quality, total square footage, and living area.\n\n"
    "Although other features like first floor area, basement size, and deck area are part of the dataset, their correlation "
    "with sale prices is generally lower than 0.4. Therefore, they play a less significant role in predicting house prices."
    )

    # Provide a link to the README for further project documentation
    st.write(
        "For more details, check out the "
        "[README file for this project](https://github.com/Patteerrik/Heritage-Housing-Issues/blob/main/README.md) "
        "to understand the full scope and purpose of the work."
    )

    # Business requirements for the project
    st.success(
        "### Business Requirements\n"
        "1. **Understand Attribute Correlations**: The client wants to understand how different features "
        "of the house affect its sale price. This will be demonstrated through visualizations of correlated variables.\n"
        "2. **Predict House Prices**: The client also wants to use these insights to predict the sale prices of four specific houses they have inherited."
    )


