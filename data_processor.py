"""
Data Processing Module for Multivariate Time Series Anomaly Detection

This module handles data loading, preprocessing, validation, and splitting
for the anomaly detection system.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
import warnings


class DataProcessor:
    """
    Handles data loading, preprocessing, and validation for time series anomaly detection.
    
    This class is responsible for:
    - Loading CSV files with timestamp validation
    - Handling missing values and data quality issues
    - Splitting data into training and analysis periods
    - Validating data requirements
    """
    
    def __init__(self, min_training_hours: int = 72):
        """
        Initialize the DataProcessor.
        
        Args:
            min_training_hours: Minimum required hours of training data (default: 72)
        """
        self.min_training_hours = min_training_hours
        self.training_start = datetime(2004, 1, 1, 0, 0)
        self.training_end = datetime(2004, 1, 5, 23, 59)
        self.analysis_end = datetime(2004, 1, 19, 7, 59)
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and validate the CSV file.
        
        Args:
            csv_path: Path to the input CSV file
            
        Returns:
            DataFrame with validated time series data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data format is invalid
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
            
        # Validate that we have data
        if df.empty:
            raise ValueError("CSV file is empty")
            
        # Check for timestamp column (assume first column is timestamp)
        if len(df.columns) < 2:
            raise ValueError("CSV must have at least 2 columns (timestamp + features)")
            
        # Convert first column to datetime if it's not already
        timestamp_col = df.columns[0]
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            raise ValueError(f"First column must be a valid timestamp: {e}")
            
        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        return df
    
    def validate_timestamps(self, df: pd.DataFrame) -> None:
        """
        Validate that timestamps are at regular intervals.
        
        Args:
            df: DataFrame with timestamp column
            
        Raises:
            ValueError: If timestamps are irregular
        """
        timestamp_col = df.columns[0]
        time_diffs = df[timestamp_col].diff().dropna()
        
        if len(time_diffs) == 0:
            return
            
        # Check if intervals are consistent (within 10% tolerance)
        median_interval = time_diffs.median()
        tolerance = median_interval * 0.1
        
        irregular_intervals = time_diffs[abs(time_diffs - median_interval) > tolerance]
        
        if len(irregular_intervals) > 0:
            warnings.warn(f"Found {len(irregular_intervals)} irregular time intervals")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward-fill and interpolation.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Handle missing values in feature columns (skip timestamp column)
        feature_cols = df_clean.columns[1:]
        
        for col in feature_cols:
            if df_clean[col].dtype in ['object', 'string']:
                # Replace non-numerical values with last good value
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Forward fill missing values
            df_clean[col] = df_clean[col].fillna(method='ffill')
            
            # If still have missing values at the beginning, use backward fill
            df_clean[col] = df_clean[col].fillna(method='bfill')
            
            # If still have missing values, use interpolation
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].interpolate(method='linear')
        
        return df_clean
    
    def handle_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle features with zero variance (constant values).
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with constant features handled
        """
        df_clean = df.copy()
        feature_cols = df_clean.columns[1:]
        
        constant_features = []
        for col in feature_cols:
            if df_clean[col].std() == 0:
                constant_features.append(col)
                # Add small noise to constant features to make them usable
                df_clean[col] = df_clean[col] + np.random.normal(0, 1e-6, len(df_clean))
        
        if constant_features:
            warnings.warn(f"Found constant features: {constant_features}. Added small noise.")
        
        return df_clean
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training (normal) and analysis periods.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Tuple of (training_data, analysis_data)
            
        Raises:
            ValueError: If insufficient training data
        """
        timestamp_col = df.columns[0]
        
        # Filter data for training period
        training_mask = (df[timestamp_col] >= self.training_start) & (df[timestamp_col] <= self.training_end)
        training_data = df[training_mask].copy()
        
        # Filter data for analysis period
        analysis_mask = (df[timestamp_col] >= self.training_start) & (df[timestamp_col] <= self.analysis_end)
        analysis_data = df[analysis_mask].copy()
        
        # Validate training data requirements
        training_hours = (training_data[timestamp_col].max() - training_data[timestamp_col].min()).total_seconds() / 3600
        
        if training_hours < self.min_training_hours:
            raise ValueError(f"Insufficient training data: {training_hours:.1f} hours < {self.min_training_hours} hours required")
        
        if len(training_data) < 10:
            raise ValueError(f"Insufficient training data points: {len(training_data)} < 10 required")
        
        return training_data, analysis_data
    
    def check_training_anomalies(self, training_data: pd.DataFrame) -> None:
        """
        Check for anomalies in training data and warn if found.
        
        Args:
            training_data: Training dataset
        """
        feature_cols = training_data.columns[1:]
        
        for col in feature_cols:
            values = training_data[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                
                if std_val > 0:
                    # Check for values beyond 3 standard deviations
                    outliers = values[abs(values - mean_val) > 3 * std_val]
                    if len(outliers) > 0:
                        warnings.warn(f"Found {len(outliers)} potential anomalies in training data for feature '{col}'")
    
    def process_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data processing pipeline.
        
        Args:
            csv_path: Path to input CSV file
            
        Returns:
            Tuple of (training_data, analysis_data)
        """
        # Load data
        df = self.load_data(csv_path)
        
        # Validate timestamps
        self.validate_timestamps(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Handle constant features
        df = self.handle_constant_features(df)
        
        # Split data
        training_data, analysis_data = self.split_data(df)
        
        # Check for training anomalies
        self.check_training_anomalies(training_data)
        
        print(f"Data processing complete:")
        print(f"  Training data: {len(training_data)} rows")
        print(f"  Analysis data: {len(analysis_data)} rows")
        print(f"  Features: {len(df.columns) - 1}")
        
        return training_data, analysis_data