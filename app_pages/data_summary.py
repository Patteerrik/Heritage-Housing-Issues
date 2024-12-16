import streamlit as st

def data_summary_body():
    # Header for the project summary
    st.write("### Project Overview")

    # Summary of the dataset used in the project
    st.info(
        "**Dataset Overview**\n"
        "This project uses data from house sales in Ames, Iowa, containing information about "
        "23 features that potentially impact the house prices. These features include aspects such as "
        "first floor area, basement size, deck area, overall quality, and construction date, among others.\n\n"
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


