# main2.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def preprocess_and_generate_data(input_file):
    """
    Preprocesses the Amazon dataset and generates new CSV files based on the proposed ERD.
    Saves all CSVs in a new folder called 'data'.
    """
    # Create the 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load the data
    df = pd.read_csv(input_file)

    # PRODUCT TABLE
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
        products_df[col] = products_df[col].astype(str).str.replace(
            '₹', '').str.replace(',', '').astype(float)

    # Clean discount_percentage
    products_df['discount_percentage'] = products_df['discount_percentage'].astype(
        str).str.replace('%', '').astype(float)

    # Clean rating
    products_df['rating'] = pd.to_numeric(
        products_df['rating'], errors='coerce')
    
    # Fill NaN ratings with a default value (e.g., 3 for neutral)
    products_df['rating'] = products_df['rating'].fillna(3)


    # Clean rating_count
    products_df['rating_count'] = products_df['rating_count'].astype(
        str).str.replace(',', '').astype(float).astype('Int64')
    median_rating_count = products_df['rating_count'].median()
    products_df['rating_count'] = products_df['rating_count'].fillna(
        median_rating_count)

    # Extract main_category
    products_df['category'] = products_df['category'].astype(str)
    products_df['category'] = products_df['category'].str.split('|').str[0]

    # Add new fields: inventory_level, reorder_point, safety_stock
    # For simplicity, let's generate values based on sales data
    products_df['inventory_level'] = np.random.randint(50, 200, size=len(products_df))
    products_df['safety_stock'] = products_df['inventory_level'] * 0.1
    products_df['reorder_point'] = products_df['safety_stock'] + (products_df['inventory_level'] * 0.3)

    # Save to CSV
    products_df.to_csv('data/Product.csv', index=False)
    print("Product data saved to data/Product.csv")

    # CUSTOMER TABLE
    # Assuming user_id and user_name correspond to customer_id and customer_name
    customer_columns = ['user_id', 'user_name']
    customers_df = df[customer_columns].drop_duplicates()
    customers_df.rename(columns={'user_id': 'customer_id', 'user_name': 'customer_name'}, inplace=True)

    # Generate synthetic regions and purchase frequency
    regions = ['North', 'South', 'East', 'West', 'Central']
    customers_df['region'] = np.random.choice(regions, size=len(customers_df))
    customers_df['purchase_frequency'] = np.random.randint(1, 10, size=len(customers_df))

    # Save to CSV
    customers_df.to_csv('data/Customer.csv', index=False)
    print("Customer data saved to data/Customer.csv")

    # REVIEW TABLE
    review_columns = ['review_id', 'product_id', 'user_id', 'rating', 'review_content']
    reviews_df = df[review_columns].drop_duplicates()
    reviews_df.rename(columns={'user_id': 'customer_id'}, inplace=True)

    # Generate review_date
    num_reviews = len(reviews_df)
    start_date = datetime(2022, 1, 1)
    reviews_df['review_date'] = [start_date + timedelta(days=int(x)) for x in np.random.randint(0, 365, size=num_reviews)]

    reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')
    reviews_df['rating'] = reviews_df['rating'].fillna(3)  # Fill NaN ratings with 3
    # Generate sentiment based on rating
    reviews_df['sentiment'] = reviews_df['rating'].apply(lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))

    # Save to CSV
    reviews_df.to_csv('data/Review.csv', index=False)
    print("Review data saved to data/Review.csv")

    # SALES TABLE
    # Generate synthetic sales data
    sales_data = []
    sales_id_counter = 1

    for idx, product in products_df.iterrows():
        product_id = product['product_id']
        price = product['discounted_price']
        start_date = datetime(2022, 1, 1)
        num_days = 365
        dates = [start_date + timedelta(days=x) for x in range(num_days)]

        for date in dates:
            # Randomly decide if a sale happened on this day
            if random.random() < 0.1:  # 10% chance of sale each day
                sales_volume = np.random.randint(1, 5)
                sales_season = 'Holiday Season' if date.month in [11, 12, 12] else 'Regular Season'
                price_at_sale = price * (1 + np.random.uniform(-0.05, 0.05))  # Small fluctuation in price

                sales_data.append({
                    'sales_id': sales_id_counter,
                    'product_id': product_id,
                    'sales_volume': sales_volume,
                    'sales_date': date.strftime('%Y-%m-%d'),
                    'sales_season': sales_season,
                    'price_at_sale': round(price_at_sale, 2)
                })
                sales_id_counter +=1

    sales_df = pd.DataFrame(sales_data)

    # Save to CSV
    sales_df.to_csv('data/Sales.csv', index=False)
    print("Sales data saved to data/Sales.csv")

    # DEMAND_FORECASTING TABLE
    # Placeholder data; actual forecasts will be generated in models2.py
    forecast_data = []

    forecast_id_counter = 1
    forecast_dates = [datetime(2023, 1, 1) + timedelta(days=7 * x) for x in range(52)]  # Weekly forecasts for one year

    for idx, product in products_df.iterrows():
        product_id = product['product_id']
        for forecast_date in forecast_dates:
            predicted_demand = np.random.randint(50, 200)
            model_used = 'Placeholder'  # Actual model used will be updated in models2.py
            forecast_data.append({
                'forecast_id': forecast_id_counter,
                'product_id': product_id,
                'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_demand': predicted_demand,
                'model_used': model_used
            })
            forecast_id_counter += 1

    forecast_df = pd.DataFrame(forecast_data)

    # Save to CSV
    forecast_df.to_csv('data/Demand_Forecasting.csv', index=False)
    print("Demand forecasting data saved to data/Demand_Forecasting.csv")


def main():
    input_file = 'amazon.csv'
    preprocess_and_generate_data(input_file)


if __name__ == "__main__":
    main()
