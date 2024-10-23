# etl.py

import pandas as pd
import numpy as np

def preprocess_amazon_sales(input_file, output_file):
    """
    Preprocesses the Amazon Sales Dataset for Looker Studio dashboard.

    Parameters:
    - input_file: Path to the input CSV file.
    - output_file: Path where the preprocessed CSV will be saved.
    """
    # Load the data
    df = pd.read_csv(input_file)
    
    # Display initial DataFrame information
    print("Initial DataFrame Info:")
    print(df.info())
    
    # Define columns to keep based on relevance
    columns_to_keep = [
        'product_id',
        'product_name',
        'category',
        'discounted_price',
        'actual_price',
        'discount_percentage',
        'rating',
        'rating_count'
    ]
    
    # Drop columns that are not needed for the dashboard
    df = df[columns_to_keep]
    print("\nColumns retained for analysis:")
    print(df.columns.tolist())
    
    # Clean 'discounted_price' and 'actual_price'
    # Remove currency symbols and commas, then convert to float
    price_columns = ['discounted_price', 'actual_price']
    for col in price_columns:
        df[col] = df[col].str.replace('₹', '').str.replace(',', '').astype(float)
        print(f"\nConverted '{col}' to float:")
        print(df[col].head())
    
    # Clean 'discount_percentage' by removing '%' and converting to float
    df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
    print("\nConverted 'discount_percentage' to float:")
    print(df['discount_percentage'].head())
    
    # Clean 'rating' by converting to float, handling invalid entries
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    print("\nConverted 'rating' to float:")
    print(df['rating'].head())
    
    # Clean 'rating_count' by removing commas and converting to integer
    df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float).astype('Int64')
    print("\nConverted 'rating_count' to integer:")
    print(df['rating_count'].head())
    
    # Extract 'main_category' from 'category' (first category before '|')
    df['main_category'] = df['category'].str.split('|').str[0]
    print("\nExtracted 'main_category':")
    print(df['main_category'].head())
    
    # Create 'discount_amount' column
    df['discount_amount'] = df['actual_price'] * (df['discount_percentage'] / 100)
    print("\nCreated 'discount_amount':")
    print(df['discount_amount'].head())
    
    # Create 'price_difference' column
    df['price_difference'] = df['actual_price'] - df['discounted_price']
    print("\nCreated 'price_difference':")
    print(df['price_difference'].head())
    
    # Handle missing values in 'rating_count' by filling with the median
    median_rating_count = df['rating_count'].median()
    df['rating_count'] = df['rating_count'].fillna(median_rating_count)
    print(f"\nFilled missing 'rating_count' with median value: {median_rating_count}")
    
    # Optional: Create 'price_category' based on 'discounted_price'
    # Categorize into 'Low', 'Medium', 'High' based on quantiles
    df['price_category'] = pd.qcut(df['discounted_price'], q=3, labels=['Low', 'Medium', 'High'])
    print("\nCreated 'price_category':")
    print(df['price_category'].head())
    
    # Optional: Extract 'brand' from 'product_name'
    # Assumes brand is the first word in 'product_name'
    df['brand'] = df['product_name'].str.split().str[0]
    print("\nExtracted 'brand' from 'product_name':")
    print(df['brand'].head())
    
    # Convert 'main_category' and 'price_category' to categorical types
    df['main_category'] = df['main_category'].astype('category')
    df['price_category'] = df['price_category'].astype('category')
    
    # Final DataFrame information
    print("\nPreprocessed DataFrame Info:")
    print(df.info())
    
    # Save the preprocessed data to the output file
    df.to_csv(output_file, index=False)
    print(f"\nPreprocessed data saved to {output_file}")

if __name__ == "__main__":
    # Specify input and output file paths
    input_file = 'amazon.csv'  # Replace with your actual input file path
    output_file = 'amazon_sales_preprocessed.csv'  # Desired output file path
    
    # Call the preprocessing function
    preprocess_amazon_sales(input_file, output_file)
