import streamlit as st


def data_summary_body():
    """Function to display the project overview and business requirements."""

    # Header for the project summary
    st.write("### Project Overview")

    # Summary of the dataset used in the project
    st.info(
        "**Dataset Overview**\n"
        "This project analyzes house sales data from Ames, Iowa, "
        "focusing on key features that impact house prices. The selected "
        "features for the prediction model are:\n"
        "- **GarageArea** (Garage size)\n"
        "- **GrLivArea** (Above-ground living area)\n"
        "- **TotalBsmtSF** (Total basement size)\n"
        "- **OverallQual** (Overall house quality, rated 1-10)\n"
        "- **YearRemodAdd** (Year of last remodel)\n\n"
        "These features were chosen based on their strong correlation (>0.5) "
        "with sale prices. They highlight the importance of space, quality, "
        "and modern renovations. Features with weaker correlations were "
        "excluded to maintain model accuracy and avoid unnecessary complexity."
    )

    # Provide a link to the README for further project documentation
    st.write(
        "For more details, check out the "
        "[README file for this project](https://github.com/Patteerrik/"
        "Heritage-Housing-Issues/blob/main/README.md) "
        "to understand the full scope and purpose of the work."
    )

    # Background and business requirements for the project
    st.success(
        "### Background Information\n"
        "Lydia Doe has inherited **four houses in Ames, Iowa** "
        "from her great-grandfather. Although she is experienced "
        "in the **Belgian** real estate market, she lacks "
        "knowledge about property values in Ames. She wants to "
        "accurately price these properties for potential sale and "
        "explore future investment opportunities.\n\n"
        "### Business Requirements\n"
        "**1. Understand Attribute Correlations**\n"
        "- Lydia wants to understand how house features (size, "
        "quality, renovations) influence sale price.\n"
        "- This will be visualized through **data analysis and "
        "correlation plots** to highlight key price drivers.\n\n"
        "**2. Predict House Prices**\n"
        "- A **machine learning model** is trained to predict "
        "house prices based on key features.\n"
        "- The model will provide **sale price estimates** for "
        "Lydia’s four inherited houses.\n"
        "- It can also predict prices for **any house in Ames**, "
        "allowing her to explore potential investments."
    )
