# Airbnb Price Prediction

## Project Overview

This project explores Airbnb listing data and develops machine learning regression models to predict listing prices based on characteristics such as location, accommodation type, availability, minimum nights, and review statistics.

The project covers the full data science workflow, including **exploratory data analysis (EDA), data preprocessing, feature engineering, model training, model evaluation, and feature importance analysis**.

## Objectives

The main objectives of this project were to:

* Explore and understand patterns within Airbnb listing data
* Clean and preprocess the dataset for machine learning
* Identify factors associated with Airbnb listing prices
* Build and compare multiple regression models
* Evaluate model performance using appropriate regression metrics
* Analyse feature importance to understand the model's predictions

## Dataset

The dataset contains **48,895 Airbnb listings** with **16 original features**, including:

* Neighbourhood group
* Room/accommodation type
* Latitude and longitude
* Price
* Minimum nights
* Number of reviews
* Reviews per month
* Host listing count
* Availability throughout the year

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the structure and distribution of the data.

The analysis included:

* Price distribution analysis
* Identification of high-priced listings and potential outliers
* Distribution of numerical features
* Comparison of accommodation types
* Comparison of listings across neighbourhood groups
* Analysis of relationships between price and other variables
* Correlation analysis using a heatmap

## Data Preprocessing

Several preprocessing and feature engineering steps were applied before model training:

* Removed unnecessary identifier and text-based columns
* Handled features containing missing values
* Renamed `room_type` to `accommodation_type` for clarity
* Applied a **log transformation** to the target price variable to reduce skewness
* Converted categorical variables into numerical features using **one-hot encoding**
* Split the data into **80% training data and 20% testing data**
* Standardised numerical features using `StandardScaler`

## Machine Learning Models

Three regression algorithms were trained and compared:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

## Model Performance

The models were evaluated using:

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **R² Score** – proportion of variation explained by the model

| Model             |       MAE |      RMSE |        R² |
| ----------------- | --------: | --------: | --------: |
| Linear Regression |     0.355 |     0.486 |     0.500 |
| Random Forest     | **0.308** | **0.433** | **0.603** |
| Gradient Boosting |     0.317 |     0.441 |     0.589 |

*The reported MAE and RMSE values are calculated on the log-transformed price target.*

### Best Model

**Random Forest Regressor** produced the strongest performance among the three models tested, achieving the lowest MAE and RMSE and the highest R² score.

The model explained approximately **60.3% of the variance in log-transformed Airbnb listing prices** in the test data.

Predictions were transformed back into the original price scale using `np.expm1()`.

## Feature Importance

Feature importance from the Random Forest model was analysed to investigate which listing characteristics contributed most to its predictions.

This provides additional interpretability beyond simply comparing model performance.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab
* Jupyter Notebook

## Skills Demonstrated

This project demonstrates practical experience with:

* Data cleaning and preprocessing
* Exploratory data analysis
* Data visualisation
* Feature engineering
* Regression modelling
* Train/test splitting
* Feature scaling
* Model comparison
* Regression evaluation metrics
* Feature importance analysis
* Machine learning using Scikit-learn

## Future Improvements

Potential improvements to the project include:

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
* Cross-validation for more robust model evaluation
* Additional feature engineering
* Investigation of outlier-handling strategies
* Testing additional regression algorithms such as XGBoost
* Further analysis of prediction errors

## Author

**Nursena Ozdemir**

Data Science MSc Student | Computing Graduate

LinkedIn: www.linkedin.com/in/nursena-ozdemir-6890b8292
GitHub:https://github.com/NursenaOzdemir001
