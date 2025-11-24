"""
missing_values_tutorial.py
Python script for handling missing values in tabular data using pandas.
Follows the steps outlined in README.md.
"""

import pandas as pd
import numpy as np

# 1. Load the sensor data
sensor_df = pd.read_csv('sensor_log.csv')
print("Loaded sensor data:")
print(sensor_df.head())

# 2. Detect and summarize missing values
print("\nMissing values per column:")
print(sensor_df.isnull().sum())

print("\nPercentage of missing values per column:")
print(sensor_df.isnull().mean() * 100)

# 3. Drop rows/columns with missing values
# Drop rows with any missing values
sensor_dropna_rows = sensor_df.dropna()
print("\nData after dropping rows with any missing values:")
print(sensor_dropna_rows.head())

# Drop columns with any missing values
sensor_dropna_cols = sensor_df.dropna(axis=1)
print("\nData after dropping columns with any missing values:")
print(sensor_dropna_cols.head())

# 4. Impute missing values using different strategies
# Fill missing values with a constant (e.g., 0)
sensor_fill_const = sensor_df.fillna(0)
print("\nData after filling missing values with 0:")
print(sensor_fill_const.head())

# Fill numeric columns with mean
sensor_fill_mean = sensor_df.fillna(sensor_df.mean(numeric_only=True))
print("\nData after filling missing values with mean:")
print(sensor_fill_mean.head())

# Fill numeric columns with median
sensor_fill_median = sensor_df.fillna(sensor_df.median(numeric_only=True))
print("\nData after filling missing values with median:")
print(sensor_fill_median.head())

# Fill categorical columns with mode
sensor_fill_mode = sensor_df.copy()
for col in sensor_fill_mode.select_dtypes(include=['object', 'category']):
    mode_val = sensor_fill_mode[col].mode(dropna=True)
    if not mode_val.empty:
        sensor_fill_mode[col].fillna(mode_val[0], inplace=True)
print("\nData after filling missing values with mode (categorical columns):")
print(sensor_fill_mode.head())

# 5. Time-series methods (forward fill, backward fill, interpolation)
# Assume data is ordered by time (if there's a time column, sort by it)
# Forward fill
sensor_ffill = sensor_df.ffill()
print("\nData after forward fill:")
print(sensor_ffill.head())

# Backward fill
sensor_bfill = sensor_df.bfill()
print("\nData after backward fill:")
print(sensor_bfill.head())

# Interpolation (numeric columns)
sensor_interp = sensor_df.interpolate()
print("\nData after interpolation:")
print(sensor_interp.head())

# 6. Compare summary statistics before and after imputation
print("\nSummary statistics before imputation:")
print(sensor_df.describe(include='all'))

print("\nSummary statistics after mean imputation:")
print(sensor_fill_mean.describe(include='all'))

# 7. Reflection (printed as comments)
"""
Reflection:
- Dropping rows/columns can result in loss of data, which may not be ideal if many values are missing.
- Filling with constants is simple but may distort analysis.
- Mean/median imputation works for numeric data, but can bias results if data is not missing at random.
- Mode imputation is suitable for categorical data.
- Time-series methods (ffill, bfill, interpolation) are useful when data is ordered by time.
- Always compare summary statistics before and after imputation to understand the impact.
"""

git clone https://github.com/Kenzie7-code/python_missingval.git