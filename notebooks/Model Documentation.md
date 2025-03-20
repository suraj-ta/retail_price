# Demand Forecasting with XGBoost model

## Model Overview
- This model leverages XGBoost, a gradient boosting algorithm, for demand forecasting.
- It predicts the `units_sold` based on historical sales data and engineered features.

### Key Components

#### 1. Feature Engineering
- **Input Features:** unit_price, competitor1 price, competitor2 price, Number of holidays in the month, Product Category name (Categorical Column).
- Encoded `product_category_name` column using a predefined mapping, Created monthly averages for **unit_price**, **customers**, and **competitor prices** per product category to capture pricing trends.

#### 2. Model Training
- Used XGBoost Regressor with optimized hyperparameters to minimize forecasting errors.
- Regularization & Optimization: Applied `L1` (reg_alpha=2) and `L2` (reg_lambda=10) **regularization** to prevent overfitting.

#### 3. Model Evaluation
- **Performance Metrics:** Assessed performance using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
- **Validation Strategy:** Assessed performance using a train-validation split, ensuring generalization to unseen data.

#### 4. Prediction
- **Future Demand Forecasting:** Predicts `units_sold` based on engineered features and past trends.
- Helps optimize inventory planning, reducing stockouts and overstock situations.

### Use Case
**Scalability:** XGBoost efficiently handles large datasets, making it ideal for real-world demand forecasting.
**Industry Applications:** Used in retail, e-commerce, and supply chain management to improve sales forecasting and optimize stock levels.