# xgboost_forecasting.py

import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def train_xgboost_model(df, product_id):
    """
    Trains an XGBoost model for a specific product.

    Parameters:
    - df: DataFrame containing the sales data.
    - product_id: The product ID to train the model on.

    Returns:
    - model: The trained XGBoost model.
    - predictions: The predicted sales.
    """
    # Filter data for the specific product
    product_data = df[df['product_id'] == product_id].sort_values('date')
    
    # Feature Engineering: Define features and target
    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'month', 'week', 'is_holiday_season']
    X = product_data[features]
    y = product_data['weekly_sales']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4, shuffle=False)
    
    # Initialize and train the model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"XGBoost Model - Product ID: {product_id}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(product_data['date'].iloc[-8:-4], y_test, label='Actual')
    plt.plot(product_data['date'].iloc[-8:-4], predictions, label='Predicted')
    plt.title(f'XGBoost Forecast for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.show()
    
    return model, predictions

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('amazon_sales_preprocessed.csv', parse_dates=['date'])
    
    # Select a sample product ID for demonstration
    sample_product_id = df['product_id'].iloc[0]
    
    # Train and forecast using XGBoost
    train_xgboost_model(df, sample_product_id)
