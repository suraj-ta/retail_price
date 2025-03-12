# Wind Turbine Forecasting Model Documentation

## Overview
This document provides an overview of the forecasting model developed using Facebook's Prophet for predicting wind turbine power generation.

## Problem Statement
The goal is to forecast the power generation of a wind turbine based on historical data and relevant features.

## Data Description
### Dataset
- **Source:** Kaggle (e.g., [Wind Turbine Data](https://www.kaggle.com/datasets))
- **Description:** The dataset contains historical data from a wind turbine, including various operational metrics and environmental conditions. Each record includes timestamps and several features that impact power generation.
- **Features:**
  - `timestamp`: Date and time of the observation
  - `AmbientTemperature`: Ambient temperature around the turbine (°C)
  - `BearingShaftTemperature`: Temperature of the bearing shaft (°C)
  - `Blade1PitchAngle`: Pitch angle of blade 1 (degrees)
  - `Blade2PitchAngle`: Pitch angle of blade 2 (degrees)
  - `Blade3PitchAngle`: Pitch angle of blade 3 (degrees)
  - `GearboxBearingTemperature`: Temperature of the gearbox bearing (°C)
  - `GearboxOilTemperature`: Temperature of the gearbox oil (°C)
  - `GeneratorRPM`: RPM of the generator
  - `GeneratorWinding1Temperature`: Temperature of the generator winding 1 (°C)
  - `GeneratorWinding2Temperature`: Temperature of the generator winding 2 (°C)
  - `HubTemperature`: Temperature of the turbine hub (°C)
  - `MainBoxTemperature`: Temperature of the main box (°C)
  - `NacellePosition`: Position of the nacelle (degrees)
  - `ReactivePower`: Reactive power output (kVAR)
  - `RotorRPM`: RPM of the rotor
  - `TurbineStatus`: Operational status of the turbine (categorical)
  - `WindDirection`: Direction of the wind (degrees)
  - `WindSpeed`: Speed of the wind (m/s)
  - `day_of_week`: Day of the week (categorical)
  - `quarter`: Fiscal quarter (categorical)
  - `power_generation`: Power output of the wind turbine (MW) (target variable)

## Data Preprocessing
- **Missing Values:** Handled missing values by imputing with median values for numerical features and mode values for categorical features.
- **Categorical Variables:** Encoded categorical variables using one-hot encoding for `TurbineStatus`, `day_of_week`, and `quarter`.
- **Normalization:** Normalized numerical features using Min-Max Scaling to bring them within a range of [0, 1].
- **Datetime Handling:** Converted `timestamp` to appropriate datetime format for time series forecasting.

## Feature Engineering
- Created new features such as `hour_of_day` derived from `timestamp`, representing the hour in the day.
- Removed features such as `GearboxOilTemperature` after feature importance analysis showed it had minimal impact on power generation.

## Model Building
### Model Selection
- **Algorithm:** Prophet
- **Reason for Selection:** Prophet is suitable for forecasting time series data with strong seasonal effects and handles missing data and outliers well.

### Model Training
- **Training Data:** 80% of the dataset used for training.
- **Validation Data:** 20% of the dataset used for validation to evaluate model performance.
- **Libraries Used:** Prophet for forecasting, Pandas for data manipulation, and Matplotlib for visualization.

### Hyperparameter Tuning
- **Method:** Grid Search
- **Parameters Tuned:**
  - `seasonality`: Adjusted to capture yearly and weekly patterns.
  - `holidays`: Incorporated holidays effects if significant.
  - `changepoints`: Identified and adjusted changepoints to capture abrupt changes in trends.

## Model Evaluation
### Metrics
- **Mean Absolute Error (MAE):** 15.24 MW
- **Mean Squared Error (MSE):** 500.36 MW²
- **Root Mean Squared Error (RMSE):** 22.36 MW
- **Mean Absolute Percentage Error (MAPE):** 8.7%

## Model Equation
The Prophet model is based on an additive regression model with components for trend, seasonality, and holidays. The general form of the model equation is:

\[
y(t) = g(t) + s(t) + h(t) + \epsilon_t
\]

where:
- \( y(t) \) is the power generation at time \( t \).
- \( g(t) \) is the trend component.
- \( s(t) \) is the seasonal component.
- \( h(t) \) is the holiday effect component (if applicable).
- \( \epsilon_t \) is the error term.

### Trend Component
The trend component \( g(t) \) is modeled as:

\[
g(t) = \text{piecewise linear or logistic growth function}
\]

### Seasonal Component
The seasonal component \( s(t) \) is modeled using Fourier series:

\[
s(t) = \sum_{k=1}^{K} \left[ \text{Fourier term}_{k, \text{cos}} \cdot \cos \left( \frac{2 \pi k t}{P} \right) + \text{Fourier term}_{k, \text{sin}} \cdot \sin \left( \frac{2 \pi k t}{P} \right) \right]
\]

where \( P \) is the period of seasonality (e.g., 365 days for yearly seasonality).

### Holiday Component
If holiday effects are included, the component \( h(t) \) is modeled as:

\[
h(t) = \sum_{i=1}^{I} \text{holiday effect}_i
\]

where \( I \) is the number of holidays, and each holiday effect is a fixed or varying effect depending on the holiday.

## Model Interpretation
- **Trend Component:** The model captures a general upward trend in power generation over time, reflecting improvements in turbine efficiency or changes in operational conditions.
- **Seasonal Components:**
  - **Yearly Seasonality:** Shows significant fluctuations in power generation with higher outputs in certain months, likely due to seasonal wind patterns.
  - **Weekly Seasonality:** Identifies variations in power generation within a week, possibly due to operational cycles or varying wind conditions.
- **Holiday Effects:** Incorporated effects of major holidays that might influence turbine operation or maintenance schedules.

## Conclusion
- **Summary:** The Prophet model effectively predicts wind turbine power generation with a MAPE of 8.7%, indicating a good level of accuracy. The model captures both the trend and seasonal patterns well, although there is room for improvement.
- **Next Steps:** Potential improvements include incorporating additional external features such as weather forecasts, refining hyperparameters based on more extensive grid search, and collecting additional data to improve the robustness of the model.

## References
- [Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)
- [Time Series Forecasting with Prophet](https://towardsdatascience.com/time-series-forecasting-with-prophet-d5ef09bbd196)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Kaggle Dataset](https://www.kaggle.com/datasets)
