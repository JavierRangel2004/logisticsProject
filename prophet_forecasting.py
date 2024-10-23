# prophet_forecasting.py

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_prophet_model(df, product_id):
    """
    Trains a Prophet model for a specific product.

    Parameters:
    - df: DataFrame containing the sales data.
    - product_id: The product ID to train the model on.

    Returns:
    - model: The trained Prophet model.
    - forecast: The forecasted values.
    """
    # Filter data for the specific product
    product_data = df[df['product_id'] == product_id].sort_values('date')
    product_data = product_data.rename(columns={'date': 'ds', 'weekly_sales': 'y'})
    
    # Split into train and test
    train = product_data.iloc[:-4]
    test = product_data.iloc[-4:]
    
    # Initialize and train Prophet model
    model = Prophet()
    model.fit(train)
    
    # Create a dataframe to hold predictions
    future = model.make_future_dataframe(periods=4, freq='W')
    forecast = model.predict(future)
    
    # Extract the forecasted values for the test period
    forecast_test = forecast.set_index('ds').loc[test['ds']]
    
    # Evaluate
    mae = mean_absolute_error(test['y'], forecast_test['yhat'])
    rmse = mean_squared_error(test['y'], forecast_test['yhat'], squared=False)
    print(f"Prophet Model - Product ID: {product_id}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Plot
    fig1 = model.plot(forecast)
    plt.title(f'Prophet Forecast for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.show()
    
    return model, forecast

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('amazon_sales_preprocessed.csv', parse_dates=['date'])
    
    # Select a sample product ID for demonstration
    sample_product_id = df['product_id'].iloc[0]
    
    # Train and forecast using Prophet
    train_prophet_model(df, sample_product_id)
