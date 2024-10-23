# main2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def preprocess_and_generate_synthetic_data(input_file, products_output, sales_output, historicSalestoGenerate):
    """
    Preprocesses the Amazon Sales Dataset, separates product attributes and sales data,
    generates synthetic time series sales data, and calculates financial metrics.

    Parameters:
    - input_file: Path to the input CSV file (amazon.csv).
    - products_output: Path where the products CSV will be saved.
    - sales_output: Path where the sales CSV with financials will be saved.
    - historicSalestoGenerate: Total number of weeks to generate.
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
        products_df[col] = products_df[col].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)

    # Clean discount_percentage
    products_df['discount_percentage'] = products_df['discount_percentage'].astype(str).str.replace('%', '').astype(float)

    # Clean rating
    products_df['rating'] = pd.to_numeric(products_df['rating'], errors='coerce')

    # Clean rating_count
    products_df['rating_count'] = products_df['rating_count'].astype(str).str.replace(',', '').astype(float).astype('Int64')
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
    periods = historicSalestoGenerate  # Total weeks to generate
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

    # Calculate Financial Metrics
    sales_df = sales_df.merge(products_df[['product_id', 'discounted_price', 'actual_price']], on='product_id', how='left')
    sales_df['earnings'] = sales_df['discounted_price'] * sales_df['weekly_sales']
    sales_df['costs'] = sales_df['actual_price'] * sales_df['weekly_sales']
    sales_df['profit'] = sales_df['earnings'] - sales_df['costs']
    sales_df['profit_margin'] = (sales_df['profit'] / sales_df['earnings']) * 100
    sales_df['profit_margin'] = sales_df['profit_margin'].replace([np.inf, -np.inf], 0).fillna(0)

    # Select relevant columns
    sales_financials_df = sales_df[['product_id', 'date', 'weekly_sales', 'earnings', 'costs', 'profit', 'profit_margin',
                                    'year', 'month', 'week', 'day_of_week', 'is_holiday_season']]

    # Save the sales dataset with financials
    sales_financials_df.to_csv(sales_output, index=False)
    print(f"\nSales data with financials saved to {sales_output}")
    print(sales_financials_df.head())

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

def main2():
    # Define Parameters
    historicSalestoGenerate = 52  # Total weeks to generate
    train_weeks = 20              # Number of weeks for training
    forecast_weeks = 4            # Number of weeks to forecast

    # Validate Parameters
    if train_weeks + forecast_weeks > historicSalestoGenerate:
        print(f"Error: train_weeks ({train_weeks}) + forecast_weeks ({forecast_weeks}) exceeds historicSalestoGenerate ({historicSalestoGenerate}).")
        return

    # Step 1: Preprocess and generate synthetic data
    input_file = 'amazon.csv'      # Replace with your actual input file path
    products_output = 'products2.csv'
    sales_output = 'sales2_financials.csv'
    # Uncomment the line below if you need to generate new synthetic data
    preprocess_and_generate_synthetic_data(input_file, products_output, sales_output, historicSalestoGenerate)

    # The rest of the steps (merging, selecting product, etc.) will be handled in models2.py

if __name__ == "__main__":
    main2()
