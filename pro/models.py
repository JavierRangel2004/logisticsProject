# models2.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import os

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def train_and_forecast():
    # Create the 'results' directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Ask the user for number of weeks to forecast
    forecast_weeks_input = input("How many weeks do you want to forecast (0 = test with sales)? ")

    try:
        forecast_weeks = int(forecast_weeks_input)
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    if forecast_weeks == 0:
        # Testing mode
        remove_weeks_input = input("How many weeks to remove from the last date? ")
        try:
            remove_weeks = int(remove_weeks_input)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            return
    else:
        remove_weeks = 0  # No weeks to remove

    # Load data
    sales_df = pd.read_csv('data/Sales.csv', parse_dates=['sales_date'])
    products_df = pd.read_csv('data/Product.csv')

    # Merge sales and products data
    merged_df = sales_df.merge(products_df, on='product_id', how='left')

    # Feature Engineering
    merged_df['month'] = merged_df['sales_date'].dt.month
    merged_df['day_of_week'] = merged_df['sales_date'].dt.dayofweek
    merged_df['week'] = merged_df['sales_date'].dt.isocalendar().week
    merged_df['is_holiday_season'] = merged_df['month'].isin([11,12,1]).astype(int)

    # Prepare data for forecasting models
    # Aggregate sales data by date
    sales_agg = merged_df.groupby('sales_date').agg({'sales_volume': 'sum'}).reset_index()

    # Time Series Forecasting
    sales_agg.set_index('sales_date', inplace=True)
    sales_ts = sales_agg['sales_volume']

    # Handle data splitting based on user input
    if remove_weeks > 0:
        # Testing mode: Remove the last 'remove_weeks' weeks for testing
        train_data = sales_ts.iloc[:-remove_weeks*7]
        test_data = sales_ts.iloc[-remove_weeks*7:]
        forecast_steps = remove_weeks * 7
    else:
        # Forecasting mode: Use all data for training
        train_data = sales_ts
        test_data = None
        forecast_steps = forecast_weeks * 7

    # Adjust frequency to daily
    train_data = train_data.asfreq('D')
    if test_data is not None:
        test_data = test_data.asfreq('D')

    # Simple Moving Average
    window = 7  # Weekly moving average
    sma_forecast, sma_errors = simple_moving_average(train_data, test_data, forecast_steps, window, "Weekly SMA", remove_weeks)
    
    # Simple Exponential Smoothing
    ses_forecast, ses_errors = simple_exponential_smoothing(train_data, test_data, forecast_steps, "SES", remove_weeks)

    # Holt's Linear Trend Method
    holt_forecast, holt_errors = holt_linear_trend(train_data, test_data, forecast_steps, "Holt's Linear Trend", remove_weeks)

    # Triple Exponential Smoothing
    seasonal_periods = 7  # Assuming weekly seasonality
    tes_forecast, tes_errors = triple_exponential_smoothing(train_data, test_data, forecast_steps, seasonal_periods, "Holt-Winters", remove_weeks)

    # ARIMA Model
    arima_forecast, arima_errors = arima_model(train_data, test_data, forecast_steps, "ARIMA", remove_weeks)

    # Prophet Model
    prophet_forecast, prophet_errors = prophet_model(train_data, test_data, forecast_steps, "Prophet", remove_weeks)

    # Save forecasts to CSV
    forecast_df = pd.DataFrame({
        'Date': sma_forecast.index,
        'SMA_Forecast': sma_forecast.values,
        'SES_Forecast': ses_forecast.values,
        "Holt's Forecast": holt_forecast.values,
        'TES_Forecast': tes_forecast.values,
        'ARIMA_Forecast': arima_forecast.values,
        'Prophet_Forecast': prophet_forecast.values
    })
    forecast_df.to_csv('results/Forecasts.csv', index=False)
    print("Forecasts saved to results/Forecasts.csv")

    # ABC Analysis
    # Calculate annual usage in units
    annual_usage = merged_df.groupby('product_id')['sales_volume'].sum().reset_index()
    # Multiply by unit cost (actual_price)
    product_costs = products_df[['product_id', 'actual_price']]
    annual_usage = annual_usage.merge(product_costs, on='product_id')
    annual_usage['annual_usage_value'] = annual_usage['sales_volume'] * annual_usage['actual_price']
    # Sort from highest to lowest
    annual_usage.sort_values(by='annual_usage_value', ascending=False, inplace=True)
    # Calculate cumulative percentage
    total_usage_value = annual_usage['annual_usage_value'].sum()
    annual_usage['cumulative_percent'] = 100 * annual_usage['annual_usage_value'].cumsum() / total_usage_value
    # Assign ABC categories
    def assign_category(cum_percent):
        if cum_percent <= 80:
            return 'A'
        elif cum_percent <= 95:
            return 'B'
        else:
            return 'C'
    annual_usage['ABC_category'] = annual_usage['cumulative_percent'].apply(assign_category)
    # Save to CSV
    annual_usage.to_csv('results/ABC_Analysis.csv', index=False)
    print("ABC analysis saved to results/ABC_Analysis.csv")

    # Inventory Analysis
    # Calculate reorder point and safety stock using basic formulas
    products_inventory = products_df[['product_id', 'inventory_level', 'safety_stock', 'reorder_point']]
    products_inventory = products_inventory.merge(annual_usage[['product_id', 'sales_volume']], on='product_id', how='left')
    products_inventory['average_daily_demand'] = products_inventory['sales_volume'] / 365
    lead_time = 7  # Assume 7 days lead time
    products_inventory['reorder_point_calculated'] = products_inventory['average_daily_demand'] * lead_time + products_inventory['safety_stock']
    # Save to CSV
    products_inventory.to_csv('results/Inventory_Analysis.csv', index=False)
    print("Inventory analysis saved to results/Inventory_Analysis.csv")

def simple_moving_average(train_data, test_data, forecast_steps, window, plot_label, remove_weeks):
    """
    Performs Simple Moving Average forecasting.
    """
    # Moving Average
    moving_avg = train_data.rolling(window=window).mean()
    last_ma = moving_avg.iloc[-1]
    forecast = [last_ma] * forecast_steps
    
    # Create date index for forecast
    if remove_weeks > 0:
        forecast_index = test_data.index
    else:
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
    
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(train_data[-60:], label='Historical Sales')
    plt.plot(forecast_series, label='Forecasted Sales', color='red')
    if remove_weeks > 0:
        plt.plot(test_data, label='Actual Sales', color='green')
    plt.title(f'Simple Moving Average Forecast ({plot_label})')
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Error Metrics
    if remove_weeks > 0:
        errors = test_data - forecast_series
        mae = mean_absolute_error(test_data, forecast_series)
        mse = mean_squared_error(test_data, forecast_series)
        mad = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / test_data)) * 100
        cfe = np.sum(errors)
        ts = cfe / mad if mad != 0 else np.nan
        print(f"Simple Moving Average - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%, TS: {ts:.2f}")
        return forecast_series, errors
    else:
        return forecast_series, None

def simple_exponential_smoothing(train_data, test_data, forecast_steps, plot_label, remove_weeks):
    """
    Performs Simple Exponential Smoothing forecasting.
    """
    model = SimpleExpSmoothing(train_data).fit()
    forecast = model.forecast(forecast_steps)
    
    # Create date index for forecast
    if remove_weeks > 0:
        forecast_index = test_data.index
    else:
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast.index = forecast_index
    
    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(train_data[-60:], label='Historical Sales')
    plt.plot(forecast, label='Forecasted Sales', color='red')
    if remove_weeks > 0:
        plt.plot(test_data, label='Actual Sales', color='green')
    plt.title(f'Simple Exponential Smoothing Forecast ({plot_label})')
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Error Metrics
    if remove_weeks > 0:
        errors = test_data - forecast
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        mad = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / test_data)) * 100
        cfe = np.sum(errors)
        ts = cfe / mad if mad != 0 else np.nan
        print(f"Simple Exponential Smoothing - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%, TS: {ts:.2f}")
        return forecast, errors
    else:
        return forecast, None

def holt_linear_trend(train_data, test_data, forecast_steps, plot_label, remove_weeks):
    """
    Performs Holt's Linear Trend Method forecasting.
    """
    model = ExponentialSmoothing(train_data, trend='add', seasonal=None).fit()
    forecast = model.forecast(forecast_steps)
    
    # Create date index for forecast
    if remove_weeks > 0:
        forecast_index = test_data.index
    else:
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast.index = forecast_index
    
    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(train_data[-60:], label='Historical Sales')
    plt.plot(forecast, label='Forecasted Sales', color='red')
    if remove_weeks > 0:
        plt.plot(test_data, label='Actual Sales', color='green')
    plt.title(f"Holt's Linear Trend Forecast ({plot_label})")
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Error Metrics
    if remove_weeks > 0:
        errors = test_data - forecast
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        mad = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / test_data)) * 100
        cfe = np.sum(errors)
        ts = cfe / mad if mad != 0 else np.nan
        print(f"Holt's Linear Trend - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%, TS: {ts:.2f}")
        return forecast, errors
    else:
        return forecast, None

def triple_exponential_smoothing(train_data, test_data, forecast_steps, seasonal_periods, plot_label, remove_weeks):
    """
    Performs Triple Exponential Smoothing (Holt-Winters) forecasting.
    """
    model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit()
    forecast = model.forecast(forecast_steps)
    
    # Create date index for forecast
    if remove_weeks > 0:
        forecast_index = test_data.index
    else:
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast.index = forecast_index
    
    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(train_data[-60:], label='Historical Sales')
    plt.plot(forecast, label='Forecasted Sales', color='red')
    if remove_weeks > 0:
        plt.plot(test_data, label='Actual Sales', color='green')
    plt.title(f'Triple Exponential Smoothing Forecast ({plot_label})')
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Error Metrics
    if remove_weeks > 0:
        errors = test_data - forecast
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        mad = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / test_data)) * 100
        cfe = np.sum(errors)
        ts = cfe / mad if mad != 0 else np.nan
        print(f"Triple Exponential Smoothing - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%, TS: {ts:.2f}")
        return forecast, errors
    else:
        return forecast, None

def arima_model(train_data, test_data, forecast_steps, plot_label, remove_weeks):
    """
    Performs ARIMA forecasting.
    """
    model = ARIMA(train_data, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    # Create date index for forecast
    if remove_weeks > 0:
        forecast_index = test_data.index
    else:
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast.index = forecast_index

    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(train_data[-60:], label='Historical Sales')
    plt.plot(forecast, label='Forecasted Sales', color='red')
    if remove_weeks > 0:
        plt.plot(test_data, label='Actual Sales', color='green')
    plt.title(f'ARIMA Forecast ({plot_label})')
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Error Metrics
    if remove_weeks > 0:
        errors = test_data - forecast
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        mad = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / test_data)) * 100
        cfe = np.sum(errors)
        ts = cfe / mad if mad != 0 else np.nan
        print(f"ARIMA Model - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%, TS: {ts:.2f}")
        return forecast, errors
    else:
        return forecast, None

def prophet_model(train_data, test_data, forecast_steps, plot_label, remove_weeks):
    """
    Performs forecasting using Facebook Prophet.
    """
    prophet_train = train_data.reset_index().rename(columns={'sales_date': 'ds', 'sales_volume': 'y'})
    model = Prophet()
    model.fit(prophet_train)
    if remove_weeks > 0:
        future = test_data.reset_index().rename(columns={'sales_date': 'ds'})
    else:
        last_date = train_data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps+1)]
        future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    forecast_series = pd.Series(forecast['yhat'].values, index=future['ds'])

    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(train_data[-60:], label='Historical Sales')
    plt.plot(forecast_series, label='Forecasted Sales', color='red')
    if remove_weeks > 0:
        plt.plot(test_data, label='Actual Sales', color='green')
    plt.title(f'Prophet Forecast ({plot_label})')
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Error Metrics
    if remove_weeks > 0:
        test_series = test_data
        errors = test_series - forecast_series
        mae = mean_absolute_error(test_series, forecast_series)
        mse = mean_squared_error(test_series, forecast_series)
        mad = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / test_series)) * 100
        cfe = np.sum(errors)
        ts = cfe / mad if mad != 0 else np.nan
        print(f"Prophet Model - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%, TS: {ts:.2f}")
        return forecast_series, errors
    else:
        return forecast_series, None

def main():
    train_and_forecast()

if __name__ == "__main__":
    main()
