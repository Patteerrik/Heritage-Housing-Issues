# Heritage Housing Issues

**Heritage Housing Issues** is a project designed to assist **Lydia Doe**, a fictional individual, in making informed decisions about selling four inherited houses in Ames, Iowa. Lydia, who has extensive real estate knowledge in Belgium, is unfamiliar with the Ames housing market. She realizes that what makes a property valuable in Belgium may not apply in Iowa, and mispricing could lead to significant financial losses.

To maximize her profits and gain a deeper understanding of property valuation in Ames, Lydia seeks help from a Data Practitioner.

## Goals

### 1. Analyze House Data  
- Identify key features that influence house prices.  
- Use correlation analysis and visualizations to highlight the most important factors.  

### 2. Predict House Prices  
- Train and optimize a Random Forest Regressor model with hyperparameter tuning.  
- Estimate prices for Lydia’s inherited houses and any other house in Ames.  

### 3. Create an Interactive Dashboard  
- Provide an easy to use Streamlit application.  
- Include feature correlation analysis, model predictions, and validation of hypotheses.  

This project delivers data driven insights and reliable price predictions to help Lydia make the best decisions when selling her properties.

The live link to the project dashboard is here: **[Heroku App Link](https://heritage-housingpp5-a255d2ccf934.herokuapp.com/)**


## Table of Contents
- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis and validation](#hypothesis-and-validation)
- [The Rationale to Map Business Requirements to Data Visualizations and ML Tasks](#the-rationale-to-map-business-requirements-to-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
  - [Data Source & Preparation](#data-source--preparation)
  - [Modeling](#modeling)
  - [Success Criteria](#success-criteria)
  - [Use Cases](#use-cases)
  - [Outcome](#outcome)
- [Epics and User Stories](#epics-and-user-stories)
- [Dashboard Design](#dashboard-design)
  - [Summary Page](#summary-page)
  - [Feature Correlation Page](#feature-correlation-page)
  - [Predict House Price Page](#predict-house-price-page)
  - [Hypotheses Validation Page](#hypotheses-validation-page)
  - [ML Model Summary Page](#ml-model-summary-page)
- [Unfixed Bugs](#unfixed-bugs)
- [Code Quality & Testing](#code-quality--testing)
  - [Manual Testing](#manual-testing)
  - [PEP8 Compliance](#pep8-compliance)
- [Deployment](#deployment)
  - [Heroku](#heroku)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
  - [Pandas](#pandas)
  - [NumPy](#numpy)
  - [Scikit-learn](#scikit-learn)
  - [Matplotlib and Seaborn](#matplotlib-and-seaborn)
  - [Streamlit](#streamlit)
- [Credits](#credits)




## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypothesis and validation

**Hypothesis 1:**

* Assumption: Bigger houses, measured by the combined area of all floors, tend to have higher sale prices compared to smaller houses.
* Validation: To validate this assumption, a correlation study will be conducted between the total area of the house and its sale price. Additionally, visualizations like scatter plots will be used to explore this relationship.

* Result:
The scatter plots and correlation analysis show a positive relationship between the total area of the house and its sale price, confirming the hypothesis that bigger houses tend to have higher sale prices.

**Hypothesis 2:**

* Assumption: Homes with higher quality ratings ('OverallQual') tend to sell for higher prices compared to homes with lower quality ratings.
* Validation: This hypothesis will be validated by conducting a correlation analysis and creating scatter plots to visualize the relationship between 'OverallQual' and 'SalePrice' to better understand how quality affects the price.

* Result: The scatter plot and correlation analysis show a strong positive link between OverallQual and SalePrice. With a correlation of 0.79, higher quality homes tend to sell for more. This confirms that home quality affects its sale price.

**Hypothesis 3:**

* Assumption: Houses with larger garage areas ('GarageArea') will sell for higher prices because buyers value bigger garages.

* Validation: To confirm this hypothesis, a correlation analysis will be performed to examine the relationship between garage size and sale price. Scatter plots will also be created to visualize how the garage size impacts the value of the house.

* Result: The scatter plot and correlation analysis show a positive relationship between 'GarageArea' and 'SalePrice'. This confirms the hypothesis that houses with bigger garages tend to sell for higher prices.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

### **Business Requirement 1**

**"The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that."**

  * To understand how different house attributes relate to the sale price, a correlation study based on Pearson correlation will be performed.
  * This helps identify which variables have the greatest impact on the sale price.
  * Scatter plots with a best-fit (regression) line will be used to clearly show the relationship between each variable and the house sale price.

### **Business Requirement 2**

**"The client is interested in predicting the house sale price from her four inherited houses, and any other house in Ames, Iowa."**

  * To predict the sale price of the four inherited houses (and other houses in Ames), a Machine Learning model will be developed to map house attributes to final prices.
  * A traditional regression model (such as Random Forest or XGBoost) will be used for this task.
  * To improve the model’s performance, hyperparameter tuning will be applied, using techniques like GridSearchCV in Scikit-Learn.


## ML Business Case

This section explains **why** and **how** we build a machine learning solution to predict house prices in Ames, Iowa.

### 1. Data Source & Preparation
- **Dataset**: A public Kaggle dataset containing ~1,500 records of houses in Ames, Iowa.  
- **Preprocessing**: We clean missing data, check for outliers, and encode categorical variables. This provides a more reliable foundation for model training.

### 2. Modeling
- **Regression Algorithms**: Since we aim to predict a continuous value (SalePrice), we will use regression methods such as Random Forest, Linear Regression.
- **Hyperparameter Tuning**: We will optimize model parameters and assess performance on both the training and test sets.

### 3. Success Criteria
- **R2 Score Requirement:** The model must achieve **≥ 0.75** on both training and test sets to be considered successful.  
- **Good Generalization**: The difference in R2 between training and test sets should be minimal to avoid overfitting.

### 4. Use Cases
- **Correlation Analysis**: Demonstrate how key features relate to sale price.
- **Price Prediction**: Estimate sale prices for the four inherited houses and any other property in Ames, ensuring the client can set an optimal and profitable price.

### 5. Outcome
- **Dashboard**: A Streamlit app that lets users visualize feature correlations and generate price predictions.
- **Client Benefit**: The client gains a reliable tool to price her inherited properties accurately and make informed decisions for future real estate opportunities.


## CRISP-DM

## Epics and User Stories

To ensure a structured approach, the project is broken down into **five epics**, each covering a key phase of development. Each epic consists of user stories that describe specific tasks required to meet business objectives.

### **Epic 1: Information Gathering and Data Collection**
- **Data Source:** Downloaded housing dataset from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data).  
- **Processing Steps:**  
  - Adjusted working directory for correct file paths.  
  - Downloaded and extracted dataset.  
  - Loaded `house_prices_records.csv` (**1460 rows, 24 columns**) for inspection.  
  - Verified `inherited_houses.csv` (**4 rows, 23 columns**) with no missing values.  
  - Converted key numerical features to `float` for consistency.  
- **Data Storage:**  
  - Saved processed dataset as `outputs/datasets/collection/HousePricesRecords.csv`.  
  - Saved inherited houses data as `outputs/datasets/collection/InheritedHouses.csv` for further use. 

### **Epic 2: Data Visualization, Cleaning, and Preparation**
- **Data Loading:** Loaded `HousePricesRecords.csv` for inspection.  
- **Missing Values Handling:**  
  - Categorical features filled with the most frequent category.  
  - Numerical features filled with the median to avoid outlier influence.  
  - Features with excessive missing values (`EnclosedPorch`, `WoodDeckSF`) were removed.  
- **Outlier Analysis:**  
  - Outliers in `LotArea`, `GrLivArea`, and `SalePrice` were identified but **not removed** to retain data integrity.  
- **Data Standardization:**  
  - Converted key numerical features to `float` for consistency.  
  - Categorical features transformed to `category` format to optimize memory.  
- **Pipeline Implementation:**  
  - Used `DropFeatures`, `CategoricalImputer`, and `MeanMedianImputer` for systematic cleaning.  
  - Ensured transformations were reproducible across datasets.  
- **Visualization:**  
  - Bar plots were created to highlight missing values.  
  - Data trends were analyzed using seaborn and matplotlib.  
- **Data Export:**  
  - Cleaned dataset saved as `HousePricesCleaned.csv` for further analysis.


### **Epic 3: Model Training, Optimization, and Validation**
- **Model Selection & Training:**  
  - Multiple models were evaluated, including `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor`, and `ExtraTreesRegressor`.  
  - Models were compared based on **R2 score**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)**.  
- **Best Model Selection:**  
  - `RandomForestRegressor` was chosen as the final model due to its strong generalization with an R2 score of **0.87** on the test set.  
- **Hyperparameter Tuning:**  
  - `GridSearchCV` was used to optimize the model by adjusting `max_depth`, `n_estimators`, `min_samples_split`, and `min_samples_leaf`.  
  - The optimized model achieved an R2 score of **0.87**, surpassing the required threshold (**≥ 0.75**).  
- **Model Validation:**  
  - **Train Set:**  
    - R2 Score: **0.961**  
    - MAE: **$9,046.78**  
    - RMSE: **$15,295.42**  
  - **Test Set:**  
    - R2 Score: **0.866**  
    - MAE: **$19,696.78**  
    - RMSE: **$30,635.80**  
- **Visualizations & Final Evaluation:**  
  - Scatter plots comparing **actual vs. predicted prices** confirmed that the model made accurate predictions.  
  - Residual plots showed minor overfitting but within acceptable limits.  
- **Final Decision:**  
  - The **optimized `RandomForestRegressor`** was saved and is ready for deployment.  
  - No further tuning is required, as the model meets all business requirements.
### **Epic 4: Planning, Design, and Development of Dashboard**
- **Framework & Technology:**  
  - The dashboard was built using **Streamlit**, allowing for interactive visualizations and model predictions.  
- **Dashboard Structure:**  
  - Multi-page navigation for different functionalities:  
    - **Data Exploration:** Visualizes key trends and distributions in the dataset.  
    - **Machine Learning Predictions:** Enables users to input house features and receive predicted prices.  
    - **Model Performance:** Displays metrics such as R2 score, MAE, and RMSE to evaluate model accuracy.  
- **Key Features:**  
  - **Dynamic Charts & Tables:** Seaborn and Matplotlib were used for data visualization.  
  - **User Input Interface:** Allows users to enter new house attributes for price prediction.  
  - **Performance Metrics Section:** Provides insight into how well the model generalizes to new data.  
- **Deployment:**  
  - The dashboard was designed to be lightweight and user-friendly, making it suitable for deployment.
### **Epic 5:** Dashboard Deployment and Release
The dashboard was deployed to **Heroku** using GitHub integration. The application is now publicly accessible for users to explore data insights and model predictions.

## Dashboard Design

### 1. Summary Page
Provides an overview of the dataset and outlines the business requirements:
- **Dataset Overview:** Lists key features selected for model training based on their correlation with sale price.
- **Business Requirements:**
  * Understanding how house features impact sale prices.
  * Predicting the sale price of inherited and other houses in Ames, Iowa.

### 2. Feature Correlation Page
Displays insights on how different house attributes correlate with the sale price:
- **Key Findings:** Highlights the most relevant features (`OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `YearRemodAdd`) based on correlation strength.
- **Visualizations:** Scatter plots and trend lines demonstrate relationships between these features and sale price.

### 3. Predict House Price Page
Allows users to predict house prices using the trained model:
- **Inherited Properties:** Lists the client's four inherited houses with their predicted sale prices.
- **New Property Prediction:** Users can input house features to get an estimated sale price.
- **Feature Constraints:** Ensures user inputs stay within valid ranges based on dataset distribution.

### 4. Hypotheses Validation Page
Tests and validates hypotheses about factors affecting house prices:
- **Hypothesis 1:** Larger houses tend to have higher sale prices.
- **Hypothesis 2:** Higher quality homes sell for higher prices.
- **Hypothesis 3:** Homes with larger garages have higher sale prices.
- **Validation Process:** Correlation analysis and scatter plots confirm these hypotheses.

### 5. ML Model Summary Page
Provides details on the machine learning model and its performance:
- **Chosen Model:** Optimized `RandomForestRegressor`.
- **Performance Metrics:**
  - **Training Set:** R2 = 0.961
  - **Test Set:** R2 = 0.866
- **Pipeline Steps:** Data preprocessing, feature selection, and hyperparameter optimization.
- **Evaluation Metrics:** Includes `Mean Squared Error (MSE)`, `Root Mean Squared Error (RMSE)`, and `Mean Absolute Error (MAE)`.


## Unfixed Bugs

There were no bugs found.
View Pipeline Details
## Testing

### Manual testing

The functionality of the application was manually tested to ensure that all key features work correctly in the deployed version. The following tests were performed:  

| **Feature**                         | **Test Description** | **Expected Result** | **Actual Result** | **Pass/Fail** |
|--------------------------------------|----------------------|----------------------|--------------------|--------------|
| **Navigation** | Click each navigation link | User is redirected to the correct page | Works as expected | ✅ Pass |
| **Data Summary Page** | Check that dataset overview is displayed | Summary statistics are shown | Works as expected | ✅ Pass |
| **Data Summary Page** | Check if readme link is working | User is directed to the correct readme page | Works as expected | ✅ Pass |
| **Feature Correlation Page** | Click correlation plots | Visualizations appear correctly | Works as expected | ✅ Pass |
| **Predict House Price** | Enter valid inputs & submit | Model predicts house price correctly | Works as expected | ✅ Pass |
| **Predict House Price** | Enter extreme values | Input validation prevents errors | Works as expected | ✅ Pass |
| **Hypotheses Validation Page** | Click to validate each hypothesis | Correct scatter plots and analysis appear | Works as expected | ✅ Pass |
| **ML Model Summary Page** | Click View Pipeline Details | Pipeline Details are displayed | Works as expected | ✅ Pass |
| **ML Model Summary Page** | Click View Performance Metrics | R2, MAE, MSE, RMSE values are displayed | Works as expected | ✅ Pass |
| **ML Model Summary Page** | Click View Predicted vs Actual Scatterplots | Train and test set results are displayed | Works as expected | ✅ Pass |

### PEP8 testing

All .py pages were tested with the [CI Python Linter](https://pep8ci.herokuapp.com/#) and passed without errors.

## Deployment

### Heroku

* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and click "new" then "create new app"
2. Giv it an "app-name" the click "create app"
3. At the Deploy tab, select GitHub as the deployment method.
4. Select your repository name and click Search. Once it is found, click "Connect".
4. Select the branch you want to deploy, then click "Deploy Branch".
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.

## Main Data Analysis and Machine Learning Libraries

### Pandas
* **Usage:** Pandas was used for data manipulation, cleaning, and transformation. For example, Pandas was used to load the housing dataset, check for missing values, drop duplicates, and convert categorical features for analysis.

### NumPy
* **Usage:** NumPy was used to handle numerical operations efficiently, such as converting data types and performing operations on arrays.

### Scikit-learn
* **Usage:** This library was essential for building and evaluating the machine learning model.
  * **Train-Test Split:** To split dataset into training and testing sets.
  * **RandomForestRegressor:** For predicting house prices based on the features provided.
  * **StandardScaler:** To scale numerical features before training the model.
  * **Pipeline:** To create a pipeline that standardizes the data and then applies the machine learning model.
  * **Evaluation Metrics:** Metrics like Mean Squared Error (MSE) and R2 were used to evaluate the model's performance.

### Matplotlib and Seaborn
* **Usage:** These libraries were used for data visualization.
  * **Matplotlib:** Was used to create different plots to understand the data, including scatter plots and line plots.
  * **Seaborn:** Was used to create correlation heatmaps and histograms to better understand the relationships between features.

### Streamlit
* **Usage**: Developing the project dashboard, which will allow users to interact with the model and visualize the results.

## Credits