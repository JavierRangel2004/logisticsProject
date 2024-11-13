# arima_forecasting.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_arima_model(df, product_id, order=(5,1,0)):
    """
    Trains an ARIMA model for a specific product.

    Parameters:
    - df: DataFrame containing the sales data.
    - product_id: The product ID to train the model on.
    - order: The (p,d,q) order of the ARIMA model.

    Returns:
    - model: The trained ARIMA model.
    - forecast: The forecasted values.
    """
    # Filter data for the specific product
    product_data = df[df['product_id'] == product_id].sort_values('date')
    product_data.set_index('date', inplace=True)
    
    # Split into train and test
    train = product_data.iloc[:-4]  # Last 4 weeks as test
    test = product_data.iloc[-4:]
    
    # Train ARIMA model
    model = ARIMA(train['weekly_sales'], order=order)
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=4)
    
    # Evaluate
    mae = mean_absolute_error(test['weekly_sales'], forecast)
    rmse = mean_squared_error(test['weekly_sales'], forecast, squared=False)
    print(f"ARIMA Model - Product ID: {product_id}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(train.index, train['weekly_sales'], label='Training')
    plt.plot(test.index, test['weekly_sales'], label='Actual')
    plt.plot(test.index, forecast, label='Forecast')
    plt.title(f'ARIMA Forecast for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.show()
    
    return model_fit, forecast

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('amazon_sales_preprocessed.csv', parse_dates=['date'])
    
    # Select a sample product ID for demonstration
    sample_product_id = df['product_id'].iloc[0]
    
    # Train and forecast using ARIMA
    train_arima_model(df, sample_product_id)
