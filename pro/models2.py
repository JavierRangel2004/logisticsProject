# models2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def extract_product_label(product_name, brand, max_words=3):
    """
    Extracts a label for the product by combining the brand and the first few words of the product name.

    Parameters:
    - product_name: The full name of the product.
    - brand: The brand of the product.
    - max_words: Maximum number of words to extract from the product name after the brand.

    Returns:
    - label: A string combining the brand and part of the product name.
    """
    # Remove the brand from the product name
    name_without_brand = product_name.replace(brand, '').strip()
    # Split the remaining name into words
    words = name_without_brand.split()
    # Take the first 'max_words' words
    extracted_words = ' '.join(words[:max_words])
    # Combine brand and extracted words
    label = f"{brand} {extracted_words}"
    return label

def train_arima_model(train_data, test_data, product_label):
    """
    Trains an ARIMA model and forecasts sales for the test period.

    Parameters:
    - train_data: Training DataFrame.
    - test_data: Testing DataFrame.
    - product_label: Label for the product for plotting.

    Returns:
    - forecast_df: DataFrame containing forecasted sales.
    """
    model = ARIMA(train_data['weekly_sales'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_data))

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': test_data['date'],
        'forecasted_sales': forecast.values
    })

    # Combine training and test data for continuous plot
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(combined_data['date'], combined_data['weekly_sales'], label='Historical Sales', marker='o',color='blue')
    plt.plot(forecast_df['date'], forecast_df['forecasted_sales'], label='ARIMA Forecast', marker='x', color='red')
    plt.title(f'ARIMA Forecast vs Actual Sales for {product_label}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate error metrics
    mae = mean_absolute_error(test_data['weekly_sales'], forecast)
    rmse = mean_squared_error(test_data['weekly_sales'], forecast, squared=False)
    print(f"ARIMA Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return forecast_df

def train_prophet_model(train_data, test_data, product_label):
    """
    Trains a Prophet model and forecasts sales for the test period.

    Parameters:
    - train_data: Training DataFrame.
    - test_data: Testing DataFrame.
    - product_label: Label for the product for plotting.

    Returns:
    - forecast_df: DataFrame containing forecasted sales.
    """
    prophet_train = train_data.rename(columns={'date': 'ds', 'weekly_sales': 'y'})
    prophet_test = test_data.rename(columns={'date': 'ds', 'weekly_sales': 'y'})

    model = Prophet()
    model.add_regressor('is_holiday_season')
    model.fit(prophet_train[['ds', 'y', 'is_holiday_season']])

    # Prepare future dataframe (test period)
    future = prophet_test[['ds', 'is_holiday_season']]
    forecast = model.predict(future)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast['ds'],
        'forecasted_sales': forecast['yhat']
    })

    # Combine training and test data for continuous plot
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(combined_data['date'], combined_data['weekly_sales'], label='Historical Sales', marker="o",color='blue')
    plt.plot(forecast_df['date'], forecast_df['forecasted_sales'], label='Prophet Forecast', marker='x', color='green')
    plt.title(f'Prophet Forecast vs Actual Sales for {product_label}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate error metrics
    mae = mean_absolute_error(prophet_test['y'], forecast['yhat'])
    rmse = mean_squared_error(prophet_test['y'], forecast['yhat'], squared=False)
    print(f"Prophet Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return forecast_df

def train_random_forest_model(train_data, test_data, product_label):
    """
    Trains a Random Forest model and forecasts sales for the test period.

    Parameters:
    - train_data: Training DataFrame.
    - test_data: Testing DataFrame.
    - product_label: Label for the product for plotting.

    Returns:
    - forecast_df: DataFrame containing forecasted sales.
    """
    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 
                'rating_count', 'month', 'week', 'is_holiday_season']
    X_train = train_data[features]
    y_train = train_data['weekly_sales']
    X_test = test_data[features]
    y_test = test_data['weekly_sales']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': test_data['date'],
        'forecasted_sales': predictions
    })

    # Combine training and test data for continuous plot
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(combined_data['date'], combined_data['weekly_sales'], label='Historical Sales', color='blue')
    plt.plot(forecast_df['date'], forecast_df['forecasted_sales'], label='Random Forest Forecast', marker='x', color='orange')
    plt.title(f'Random Forest Forecast vs Actual Sales for {product_label}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate error metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Random Forest Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return forecast_df

def train_xgboost_model(train_data, test_data, product_label):
    """
    Trains an XGBoost model and forecasts sales for the test period.

    Parameters:
    - train_data: Training DataFrame.
    - test_data: Testing DataFrame.
    - product_label: Label for the product for plotting.

    Returns:
    - forecast_df: DataFrame containing forecasted sales.
    """
    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 
                'rating_count', 'month', 'week', 'is_holiday_season']
    X_train = train_data[features]
    y_train = train_data['weekly_sales']
    X_test = test_data[features]
    y_test = test_data['weekly_sales']

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': test_data['date'],
        'forecasted_sales': predictions
    })

    # Combine training and test data for continuous plot
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(combined_data['date'], combined_data['weekly_sales'], label='Historical Sales', color='blue')
    plt.plot(forecast_df['date'], forecast_df['forecasted_sales'], label='XGBoost Forecast', marker='x', color='purple')
    plt.title(f'XGBoost Forecast vs Actual Sales for {product_label}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate error metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"XGBoost Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return forecast_df

def main():
    # Define Parameters
    historicSalestoGenerate = 52  # Total weeks to generate
    train_weeks = 20              # Number of weeks for training
    forecast_weeks = 4            # Number of weeks to forecast

    # Validate Parameters
    if train_weeks + forecast_weeks > historicSalestoGenerate:
        print(f"Error: train_weeks ({train_weeks}) + forecast_weeks ({forecast_weeks}) exceeds historicSalestoGenerate ({historicSalestoGenerate}).")
        return

    # File paths
    products_output = 'products2.csv'
    sales_output = 'sales2_financials.csv'

    # Load the preprocessed data with financials
    sales_financials_df = pd.read_csv(sales_output, parse_dates=['date'])

    # Merge with product data
    products_df = pd.read_csv(products_output)
    merged_df = sales_financials_df.merge(products_df, on='product_id', how='left')
    print("\nMerged DataFrame Info:")
    print(merged_df.info())

    # Check for any missing values after merge
    missing_values = merged_df.isnull().sum()
    print("\nMissing Values After Merge:")
    print(missing_values)

    # Handle any missing values if necessary
    # For this example, we'll drop any rows with missing values
    merged_df = merged_df.dropna()
    print("\nDataFrame after dropping missing values:")
    print(merged_df.info())

    # Select a sample product ID for demonstration
    sample_index = 100  # Change the index as needed to select different products
    unique_products = merged_df['product_id'].unique()
    if sample_index >= len(unique_products):
        print("Sample index out of range. Please choose a smaller index.")
        return

    sample_product_id = unique_products[sample_index]
    sample_product = merged_df[merged_df['product_id'] == sample_product_id].iloc[0]
    product_label = extract_product_label(sample_product['product_name'], sample_product['brand'])
    print(f"\nSelected Product for Forecasting: {product_label} (Product ID: {sample_product_id})")

    # Prepare training and testing data
    product_data = merged_df[merged_df['product_id'] == sample_product_id].sort_values('date')

    # Define training period and testing period
    train_data = product_data.iloc[:train_weeks]
    test_data = product_data.iloc[train_weeks:train_weeks + forecast_weeks]

    print(f"\nTraining Data: {len(train_data)} weeks")
    print(train_data[['date', 'weekly_sales']].head())
    print(f"\nTesting Data: {len(test_data)} weeks")
    print(test_data[['date', 'weekly_sales']].head())

    # Train and forecast using models
    print("\n--- ARIMA Model ---")
    arima_forecast = train_arima_model(train_data, test_data, product_label)

    print("\n--- Prophet Model ---")
    prophet_forecast = train_prophet_model(train_data, test_data, product_label)

    print("\n--- Random Forest Model ---")
    rf_forecast = train_random_forest_model(train_data, test_data, product_label)

    print("\n--- XGBoost Model ---")
    xgb_forecast = train_xgboost_model(train_data, test_data, product_label)

if __name__ == "__main__":
    main()
