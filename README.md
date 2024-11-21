# Heritage Housing Issues

## Table of Contents
- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis and Validation](#hypothesis-and-validation)
- [The Rationale to Map Business Requirements to Data Visualizations and ML Tasks](#the-rationale-to-map-business-requirements-to-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Dashboard Design](#dashboard-design)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
  - [Heroku](#heroku)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
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

* Houses with larger garage areas ('GarageArea') will sell for higher prices because buyers value bigger garages.

* Validation: To confirm this hypothesis, a correlation analysis will be performed to examine the relationship between garage size and sale price. Scatter plots will also be created to visualize how the garage size impacts the value of the house.

* Result: The scatter plot and correlation analysis show a positive relationship between 'GarageArea' and 'SalePrice'. This confirms the hypothesis that houses with bigger garages tend to sell for higher prices.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

* **Business Requirement 1**

  * To understand how different house attributes relate to the sale price, we will perform a correlation study or use a Predictive Power Score (PPS) analysis.
  * This will help us identify the variables most significantly impacting the sale price.
  * We will visualize these variables against the sale price to derive insights.

* **Business Requirement 2**

  * To predict the total sale price of the four inherited houses, we will create a Machine Learning (ML) model that can map the relationships between house features and the sale price.
  * We can use either traditional ML models or Neural Networks.
  * To enhance the model's performance, we will conduct hyperparameter optimization using tools like Scikit-Learn.

## ML Business Case

The client wants to understand what affects house prices and predict the prices of specific houses, including four inherited houses. To achieve this, we need to use a machine learning model.

The ML task is to build a model that can predict house prices based on different features of the house, like the area, number of rooms, year built, etc. Since the price is a number, a regression model is a good choice.

Key Points for the ML Model:
* **Inputs:** Information about the house, such as the number of rooms, lot size, garage area, and more.
* **Output:** The predicted price of the house.
* **Model Type:** We will use a regression model, like Random Forest, Linear Regression, or Neural Networks, to find the relationship between house features and sale prices.
* **Success Criteria:** The client considers the model successful if it has an R² score of at least 0.75. This means the model should be able to explain at least 75% of the changes in house prices.
* **Use Cases:**
  * Show how different house features are related to sale prices.
  * Predict the sale prices of the four inherited houses and other houses in Ames, Iowa.
The goal is to give the client a reliable tool to estimate house prices, helping them get the best value for their inherited houses and make good decisions for other properties.

## Epics and User Stories

* **Epic 1:** Information Gathering and Data Collection
  - User Story: Collect relevant data and information for analysis.
* **Epic 2:** Data Visualization, Cleaning, and Preparation
  - User Story: Visualize the data, identify trends, and clean the dataset.
* **Epic 3:** Model Training, Optimization, and Validation
  - User Story: Train the model, tune hyperparameters, and validate model performance.
* **Epic 4:** Planning, Design, and Development of Dashboard
  - User Story: Design and develop an interactive dashboard to visualize insights.
* **Epic 5:** Dashboard Deployment and Release
  - User Story: Deploy the dashboard to Heroku and make it publicly accessible.

## Dashboard Design

1. **Summary Page:** Provides an overview of the dataset and states the business requirements.
2. **Feature Correlation Page:** Lists findings related to features most correlated with sales price.
3. **Inherited Houses Page:** Displays the four inherited houses and their respective predicted sale prices, including the summed sale price.
4. **Hypotheses Page:** Describes the hypotheses and how they were validated.
5. **Technical Page:** Shows the model's performance and pipeline steps.

## Unfixed Bugs


## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

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
  * **Evaluation Metrics:** Metrics like Mean Squared Error (MSE) and R² were used to evaluate the model's performance.

### Matplotlib and Seaborn
* **Usage:** These libraries were used for data visualization.
  * **Matplotlib:** Was used to create different plots to understand the data, including scatter plots and line plots.
  * **Seaborn:** Was used to create correlation heatmaps and histograms to better understand the relationships between features.

### Feature-engine
* **Usage**: Using `SmartCorrelatedSelection` from the Feature-engine library to select relevant features and reduce multicollinearity among predictors.

### Streamlit (planned)
* **Usage**: Developing the project dashboard, which will allow users to interact with the model and visualize the results.

### Content

## Credits