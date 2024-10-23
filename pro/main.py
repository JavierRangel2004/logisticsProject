# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def preprocess_and_generate_synthetic_data(input_file, products_output, sales_output):
    """
    Preprocesses the Amazon Sales Dataset, separates product attributes and sales data,
    and generates synthetic time series sales data.

    Parameters:
    - input_file: Path to the input CSV file (amazon.csv).
    - products_output: Path where the products CSV will be saved.
    - sales_output: Path where the sales CSV will be saved.
    """
    # Load the data
    df = pd.read_csv(input_file)

    # Define columns to keep for products
    product_columns = [
        'product_id',
        'product_name',
        'category',
        'discounted_price',
        'actual_price',
        'discount_percentage',
        'rating',
        'rating_count'
    ]

    # Extract product attributes
    products_df = df[product_columns].drop_duplicates()

    # Clean price columns
    price_columns = ['discounted_price', 'actual_price']
    for col in price_columns:
        products_df[col] = products_df[col].str.replace('₹', '').str.replace(',', '').astype(float)

    # Clean discount_percentage
    products_df['discount_percentage'] = products_df['discount_percentage'].str.replace('%', '').astype(float)

    # Clean rating
    products_df['rating'] = pd.to_numeric(products_df['rating'], errors='coerce')

    # Clean rating_count
    products_df['rating_count'] = products_df['rating_count'].str.replace(',', '').astype(float).astype('Int64')
    median_rating_count = products_df['rating_count'].median()
    products_df['rating_count'] = products_df['rating_count'].fillna(median_rating_count)

    # Extract main_category
    products_df['main_category'] = products_df['category'].str.split('|').str[0]

    # Extract brand from product_name (assuming brand is the first word)
    products_df['brand'] = products_df['product_name'].str.split().str[0]

    # Select final product columns
    products_df = products_df[['product_id', 'product_name', 'main_category', 'brand',
                               'discounted_price', 'actual_price', 'discount_percentage',
                               'rating', 'rating_count']]

    # Convert to categorical types where appropriate
    products_df['main_category'] = products_df['main_category'].astype('category')
    products_df['brand'] = products_df['brand'].astype('category')

    # Save the products dataset
    products_df.to_csv(products_output, index=False)
    print(f"\nProducts data saved to {products_output}")
    print(products_df.head())

    # Generate synthetic sales data
    # Define time range
    start_date = datetime(2023, 1, 1)
    periods = 52  # Weekly data for one year
    freq = 'W'  # Weekly frequency

    synthetic_sales = []

    for _, product in products_df.iterrows():
        product_id = product['product_id']
        main_category = product['main_category']
        brand = product['brand']
        price = product['discounted_price']
        rating = product['rating']
        rating_count = product['rating_count']

        # Simulate sales over time with some randomness
        for week in range(periods):
            date = start_date + timedelta(weeks=week)
            # Base sales influenced by price and rating
            base = max(1, 100 - (price / 10) + (rating * 10))
            # Introduce seasonality (e.g., higher sales in certain weeks)
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * week / periods)
            # Random fluctuation
            random_factor = np.random.normal(1.0, 0.1)
            weekly_sales = max(0, int(base * seasonal * random_factor))
            synthetic_sales.append({
                'product_id': product_id,
                'date': date.strftime('%Y-%m-%d'),
                'weekly_sales': weekly_sales
            })

    sales_df = pd.DataFrame(synthetic_sales)

    # Feature Engineering
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    sales_df['year'] = sales_df['date'].dt.year
    sales_df['month'] = sales_df['date'].dt.month
    sales_df['week'] = sales_df['date'].dt.isocalendar().week
    sales_df['day_of_week'] = sales_df['date'].dt.dayofweek
    sales_df['is_holiday_season'] = sales_df['month'].isin([11, 12, 1, 2])  # Example: Nov-Feb as holiday season

    # Save the sales dataset
    sales_df.to_csv(sales_output, index=False)
    print(f"\nSales data saved to {sales_output}")
    print(sales_df.head())

def train_arima_model(merged_df, product_id, order=(5,1,0)):
    """
    Trains an ARIMA model for a specific product.

    Parameters:
    - merged_df: DataFrame containing the merged sales and product data.
    - product_id: The product ID to train the model on.
    - order: The (p,d,q) order of the ARIMA model.

    Returns:
    - model_fit: The trained ARIMA model.
    - forecast: The forecasted values.
    """
    # Filter data for the specific product
    product_data = merged_df[merged_df['product_id'] == product_id].sort_values('date')
    product_data.set_index('date', inplace=True)

    # Ensure the index is datetime
    product_data.index = pd.to_datetime(product_data.index)

    # Train on the entire year
    train = product_data.copy()

    # Forecast the next month (4 weeks)
    model = ARIMA(train['weekly_sales'], order=order)
    model_fit = model.fit()

    # Forecast
    forecast_steps = 4
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create future dates for plotting
    last_date = product_data.index[-1]
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_steps + 1)]

    # Convert forecast to a DataFrame for plotting
    forecast_df = pd.DataFrame({'weekly_sales': forecast.values}, index=future_dates)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(product_data.index, product_data['weekly_sales'], label='Historical Sales')
    plt.plot(forecast_df.index, forecast_df['weekly_sales'], label='Forecasted Sales', marker='o')
    plt.title(f'ARIMA Forecast for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Since we don't have actual sales for the forecasted period, we can't compute MAE and RMSE
    print(f"ARIMA Model - Product ID: {product_id}, Forecasted next {forecast_steps} weeks.")

    return model_fit, forecast_df

def train_prophet_model(merged_df, product_id):
    """
    Trains a Prophet model for a specific product.

    Parameters:
    - merged_df: DataFrame containing the merged sales and product data.
    - product_id: The product ID to train the model on.

    Returns:
    - model: The trained Prophet model.
    - forecast: The forecasted values.
    """
    # Filter data for the specific product
    product_data = merged_df[merged_df['product_id'] == product_id].sort_values('date')
    product_data = product_data.rename(columns={'date': 'ds', 'weekly_sales': 'y'})

    # Initialize and train Prophet model with additional regressor
    model = Prophet()
    model.add_regressor('is_holiday_season')
    model.fit(product_data[['ds', 'y', 'is_holiday_season']])

    # Forecast the next month (4 weeks)
    forecast_steps = 4
    future = model.make_future_dataframe(periods=forecast_steps, freq='W')

    # For future regressor values, assume 'is_holiday_season' based on month
    future['month'] = future['ds'].dt.month
    future['is_holiday_season'] = future['month'].isin([11, 12, 1, 2])

    # Predict
    forecast = model.predict(future)

    # Extract forecasted values
    forecast_df = forecast[['ds', 'yhat']].tail(forecast_steps)

    # Plot
    fig1 = model.plot(forecast)
    plt.title(f'Prophet Forecast for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.show()

    # Since we don't have actual sales for the forecasted period, we can't compute MAE and RMSE
    print(f"Prophet Model - Product ID: {product_id}, Forecasted next {forecast_steps} weeks.")

    return model, forecast_df

def train_random_forest(merged_df, product_id):
    """
    Trains a Random Forest model for a specific product.

    Parameters:
    - merged_df: DataFrame containing the merged sales and product data.
    - product_id: The product ID to train the model on.

    Returns:
    - model: The trained Random Forest model.
    - predictions: The predicted sales.
    """
    # Filter data for the specific product
    product_data = merged_df[merged_df['product_id'] == product_id].sort_values('date')

    # Feature Engineering: Define features and target
    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'month', 'week', 'is_holiday_season']
    X = product_data[features]
    y = product_data['weekly_sales']

    # Define the training set (entire year)
    X_train = X
    y_train = y

    # Define the forecast period (next 4 weeks)
    forecast_steps = 4
    last_date = product_data['date'].max()
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_steps + 1)]
    
    # Create synthetic future data for prediction
    # Here, we assume that discount, price, rating remain the same, and set 'is_holiday_season' based on month
    last_product = product_data.iloc[-1]
    future_data = pd.DataFrame({
        'discounted_price': [last_product['discounted_price']] * forecast_steps,
        'actual_price': [last_product['actual_price']] * forecast_steps,
        'discount_percentage': [last_product['discount_percentage']] * forecast_steps,
        'rating': [last_product['rating']] * forecast_steps,
        'rating_count': [last_product['rating_count']] * forecast_steps,
        'month': [date.month for date in future_dates],
        'week': [date.isocalendar().week for date in future_dates],
        'is_holiday_season': [date.month in [11, 12, 1, 2] for date in future_dates]
    })

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(future_data)

    # Create a DataFrame for forecasted sales
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'weekly_sales': predictions
    })

    # Plot historical and forecasted sales
    plt.figure(figsize=(12,6))
    plt.plot(product_data['date'], product_data['weekly_sales'], label='Historical Sales')
    plt.plot(forecast_df['date'], forecast_df['weekly_sales'], label='Forecasted Sales', marker='o')
    plt.title(f'Random Forest Forecast for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Since we don't have actual sales for the forecasted period, we can't compute MAE and RMSE
    print(f"Random Forest Model - Product ID: {product_id}, Forecasted next {forecast_steps} weeks.")

    return model, forecast_df

def train_xgboost_model(merged_df, product_id):
    """
    Trains an XGBoost model for a specific product.

    Parameters:
    - merged_df: DataFrame containing the merged sales and product data.
    - product_id: The product ID to train the model on.

    Returns:
    - model: The trained XGBoost model.
    - predictions: The predicted sales.
    """
    from xgboost import XGBRegressor

    # Filter data for the specific product
    product_data = merged_df[merged_df['product_id'] == product_id].sort_values('date')

    # Feature Engineering: Define features and target
    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'month', 'week', 'is_holiday_season']
    X = product_data[features]
    y = product_data['weekly_sales']

    # Define the training set (entire year)
    X_train = X
    y_train = y

    # Define the forecast period (next 4 weeks)
    forecast_steps = 4
    last_date = product_data['date'].max()
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_steps + 1)]
    
    # Create synthetic future data for prediction
    # Here, we assume that discount, price, rating remain the same, and set 'is_holiday_season' based on month
    last_product = product_data.iloc[-1]
    future_data = pd.DataFrame({
        'discounted_price': [last_product['discounted_price']] * forecast_steps,
        'actual_price': [last_product['actual_price']] * forecast_steps,
        'discount_percentage': [last_product['discount_percentage']] * forecast_steps,
        'rating': [last_product['rating']] * forecast_steps,
        'rating_count': [last_product['rating_count']] * forecast_steps,
        'month': [date.month for date in future_dates],
        'week': [date.isocalendar().week for date in future_dates],
        'is_holiday_season': [date.month in [11, 12, 1, 2] for date in future_dates]
    })

    # Initialize and train the model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(future_data)

    # Create a DataFrame for forecasted sales
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'weekly_sales': predictions
    })

    # Plot historical and forecasted sales
    plt.figure(figsize=(12,6))
    plt.plot(product_data['date'], product_data['weekly_sales'], label='Historical Sales')
    plt.plot(forecast_df['date'], forecast_df['weekly_sales'], label='Forecasted Sales', marker='o')
    plt.title(f'XGBoost Forecast for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Since we don't have actual sales for the forecasted period, we can't compute MAE and RMSE
    print(f"XGBoost Model - Product ID: {product_id}, Forecasted next {forecast_steps} weeks.")

    return model, forecast_df

def main():
    # Step 1: Preprocess and generate synthetic data
    input_file = 'amazon.csv'  # Replace with your actual input file path
    products_output = 'products.csv'
    sales_output = 'sales.csv'
    preprocess_and_generate_synthetic_data(input_file, products_output, sales_output)

    # Step 2: Load the preprocessed data with proper date parsing
    products_df = pd.read_csv(products_output)
    sales_df = pd.read_csv(sales_output, parse_dates=['date'])

    # Step 3: Merge datasets on 'product_id'
    merged_df = pd.merge(sales_df, products_df, on='product_id', how='left')
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

    # Step 4: Select a sample product ID for demonstration
    sample_product_id = merged_df['product_id'].iloc[100]
    print(f"\nSelected Product ID for Forecasting: {sample_product_id}")

    # Step 5: Train and forecast using ARIMA
    train_arima_model(merged_df, sample_product_id)

    # Step 6: Train and forecast using Prophet
    train_prophet_model(merged_df, sample_product_id)

    # Step 7: Train and forecast using Random Forest
    train_random_forest(merged_df, sample_product_id)

    # Step 8: Train and forecast using XGBoost
    train_xgboost_model(merged_df, sample_product_id)

if __name__ == "__main__":
    main()
