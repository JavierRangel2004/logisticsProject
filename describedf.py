import pandas as pd

# Load the CSV file into a DataFrame amazon_sales_preprocessed.csv
df = pd.read_csv('amazon_sales_preprocessed.csv')

# Display basic information about the DataFrame
print("Basic Information:")
print(df.info())

# Display the first few rows of the DataFrame
print("\nFirst 5 Rows:")
print(df.head())

# Display summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Display the number of missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# Display the data types of each column
print("\nData Types:")
print(df.dtypes)

# Display the unique values and their counts for each categorical column
print("\nUnique Values in Categorical Columns:")
for column in df.select_dtypes(include=['object']).columns:
    print(f"\n{column}:")
    print(df[column].value_counts())

# Display correlation matrix for numerical columns
print("\nCorrelation Matrix:")
print(df.corr())