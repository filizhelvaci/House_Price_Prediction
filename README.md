# House Price Prediction Project

This project focuses on predicting house prices based on various features of the property. We use a machine learning model to predict the `SalePrice` of properties.

## Project Overview

The dataset used for this project contains multiple features related to the properties, including construction details, location, and quality attributes. Our goal is to predict the sale price (`SalePrice`) of a property.

### Dataset Description

The dataset consists of various columns such as:

- `SalePrice`: The sale price of the property (target variable).
- `MSSubClass`: The construction class of the property.
- `MSZoning`: General zoning classification of the property.
- `LotFrontage`: The length of the property’s frontage on the street.
- `LotArea`: The size of the property (in square feet).
- `Street`: The type of street access.
- `Alley`: The type of alley access to the property.
- `LotShape`: The general shape and condition of the property.
- `LandContour`: The flatness or contour of the land.
- `Neighborhood`: The neighborhood where the property is located.
- `OverallQual`: The overall material and finish quality of the house.
- `YearBuilt`: The year the house was built.
- ... (Other features describe various aspects like foundation, basement, heating, and more).

## Data Preprocessing & Feature Engineering

- **Missing Value Handling**: Missing values in several columns are handled, and some columns are filled with default values or median values.
- **Outliers**: Outliers in numerical variables are replaced with appropriate thresholds.
- **Feature Engineering**: Several new features are created, such as:
  - `NEW_LAND_CONTOUR`: Encodes the property’s contour.
  - `NEW_YEAR_BUILT`: Encodes different periods of house construction.
  - `NEW_BSMT_VALUE`: Combines multiple basement-related features into a single value.
  - `NEW_FIREPLACES`: Combines features related to fireplaces.

## EDA (Exploratory Data Analysis)

During the exploratory data analysis (EDA), we examined the relationships between the target variable (`SalePrice`) and independent variables. Correlation analysis was performed, and irrelevant or highly correlated features were removed to improve model performance.

## Models

Several machine learning models are used in this project, including:

- **LightGBM Regressor**
- **Random Forest Regressor**

The models are tuned using hyperparameter optimization techniques such as GridSearchCV.

## Key Findings

- The `SalePrice` is highly correlated with features like `OverallQual`, `GrLivArea`, `TotRmsAbvGrd`, and others.
- Feature engineering plays a crucial role in improving model performance, particularly with the creation of new features based on domain knowledge.

## Requirements

- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- LightGBM

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
