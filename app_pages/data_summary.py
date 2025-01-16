import streamlit as st

def data_summary_body():

    # Header for the project summary
    st.write("### Project Overview")

    # Summary of the dataset used in the project
    st.info(
    "**Dataset Overview**\n"
    "This project uses data from house sales in Ames, Iowa, containing information about "
    "various features that potentially impact house prices. The selected features used in the prediction model "
    "include 'GarageArea', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', and 'YearRemodAdd', all of which "
    "have a correlation greater than 0.5 with the sale price. These features reflect critical aspects such as "
    "the overall quality of the house, the size of the living areas, and the age of recent remodels.\n\n"
    "These features were chosen for their strong relationship with house prices, emphasizing the importance "
    "of quality, space, and modern updates in the housing market. Other potential features, such as "
    "first floor area or deck size, were not included as their correlations with sale prices are generally "
    "lower and less impactful for this model."
)


    # Provide a link to the README for further project documentation
    st.write(
        "For more details, check out the "
        "[README file for this project](https://github.com/Patteerrik/Heritage-Housing-Issues/blob/main/README.md) "
        "to understand the full scope and purpose of the work."
    )

    # Background and business requirements for the project
    st.success(
        "### Background Information\n"
        "Lydia Doe has inherited four houses in Ames, Iowa from her great-grandfather. "
        "While knowledgeable about property prices in Belgium, she's unsure about the Iowa market. "
        "Lydia needs to accurately price these houses to maximize her returns and is also interested in "
        "the general property market in Ames for potential future investments.\n\n"
        "### Business Requirements\n"
        "1. **Understand Attribute Correlations**: The client wants to understand how different features "
        "of the house affect its sale price. This will be demonstrated through visualizations of correlated variables.\n"
        "2. **Predict House Prices**: The client also wants to use these insights to predict the sale prices of four specific houses they have inherited."
    )





