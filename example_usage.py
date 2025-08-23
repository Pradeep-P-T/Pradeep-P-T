"""
Example Usage Script for Multivariate Time Series Anomaly Detection

This script demonstrates how to use the anomaly detection system
with different methods and parameters.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from main import main
from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector
from feature_attributor import FeatureAttributor


def create_sample_data(output_path: str = "sample_data.csv", n_rows: int = 1000) -> str:
    """
    Create sample time series data for testing.
    
    Args:
        output_path: Path to save the sample data
        n_rows: Number of rows to generate
        
    Returns:
        Path to the created sample data file
    """
    print("Creating sample data...")
    
    # Generate timestamps
    start_time = datetime(2004, 1, 1, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_rows)]
    
    # Generate normal features
    np.random.seed(42)  # For reproducibility
    
    # Feature 1: Temperature (normal range 20-30°C)
    temp_normal = np.random.normal(25, 2, n_rows)
    
    # Feature 2: Pressure (normal range 100-120 kPa)
    pressure_normal = np.random.normal(110, 5, n_rows)
    
    # Feature 3: Humidity (normal range 40-60%)
    humidity_normal = np.random.normal(50, 5, n_rows)
    
    # Feature 4: Flow rate (normal range 10-15 L/min)
    flow_normal = np.random.normal(12.5, 1, n_rows)
    
    # Feature 5: Vibration (normal range 0.1-0.5 mm/s)
    vibration_normal = np.random.normal(0.3, 0.1, n_rows)
    
    # Add some anomalies
    anomaly_indices = [200, 400, 600, 800]  # Introduce anomalies at these indices
    
    for idx in anomaly_indices:
        if idx < n_rows:
            # Temperature spike
            temp_normal[idx] = np.random.normal(40, 2)
            # Pressure drop
            pressure_normal[idx] = np.random.normal(80, 5)
            # Humidity anomaly
            humidity_normal[idx] = np.random.normal(80, 5)
            # Flow rate anomaly
            flow_normal[idx] = np.random.normal(20, 2)
            # Vibration spike
            vibration_normal[idx] = np.random.normal(1.5, 0.2)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temp_normal,
        'pressure': pressure_normal,
        'humidity': humidity_normal,
        'flow_rate': flow_normal,
        'vibration': vibration_normal
    })
    
    # Save to CSV
    data.to_csv(output_path, index=False)
    print(f"Sample data created: {output_path}")
    print(f"Shape: {data.shape}")
    print(f"Features: {list(data.columns[1:])}")
    
    return output_path


def run_basic_example():
    """Run basic example with default settings."""
    print("\n" + "="*60)
    print("BASIC EXAMPLE - Isolation Forest")
    print("="*60)
    
    # Create sample data
    input_file = create_sample_data("sample_basic.csv", 500)
    output_file = "output_basic.csv"
    
    # Run anomaly detection
    main(
        input_csv_path=input_file,
        output_csv_path=output_file,
        method="isolation_forest"
    )
    
    # Display results
    results = pd.read_csv(output_file)
    print(f"\nResults summary:")
    print(f"Total rows: {len(results)}")
    print(f"Mean abnormality score: {results['Abnormality_score'].mean():.2f}")
    print(f"Max abnormality score: {results['Abnormality_score'].max():.2f}")
    
    # Show top anomalies
    top_anomalies = results.nlargest(5, 'Abnormality_score')
    print(f"\nTop 5 anomalies:")
    for idx, row in top_anomalies.iterrows():
        print(f"Row {idx}: Score {row['Abnormality_score']:.2f}, "
              f"Top features: {row['top_feature_1']}, {row['top_feature_2']}")


def run_autoencoder_example():
    """Run example with autoencoder method."""
    print("\n" + "="*60)
    print("AUTOENCODER EXAMPLE")
    print("="*60)
    
    # Create sample data
    input_file = create_sample_data("sample_autoencoder.csv", 300)
    output_file = "output_autoencoder.csv"
    
    # Run anomaly detection with autoencoder
    main(
        input_csv_path=input_file,
        output_csv_path=output_file,
        method="autoencoder",
        epochs=30,
        batch_size=16
    )
    
    # Display results
    results = pd.read_csv(output_file)
    print(f"\nResults summary:")
    print(f"Total rows: {len(results)}")
    print(f"Mean abnormality score: {results['Abnormality_score'].mean():.2f}")
    print(f"Max abnormality score: {results['Abnormality_score'].max():.2f}")


def run_ensemble_example():
    """Run example with ensemble method."""
    print("\n" + "="*60)
    print("ENSEMBLE EXAMPLE")
    print("="*60)
    
    # Create sample data
    input_file = create_sample_data("sample_ensemble.csv", 400)
    output_file = "output_ensemble.csv"
    
    # Run anomaly detection with ensemble
    main(
        input_csv_path=input_file,
        output_csv_path=output_file,
        method="ensemble"
    )
    
    # Display results
    results = pd.read_csv(output_file)
    print(f"\nResults summary:")
    print(f"Total rows: {len(results)}")
    print(f"Mean abnormality score: {results['Abnormality_score'].mean():.2f}")
    print(f"Max abnormality score: {results['Abnormality_score'].max():.2f}")


def run_custom_parameters_example():
    """Run example with custom parameters."""
    print("\n" + "="*60)
    print("CUSTOM PARAMETERS EXAMPLE")
    print("="*60)
    
    # Create sample data
    input_file = create_sample_data("sample_custom.csv", 600)
    output_file = "output_custom.csv"
    
    # Run anomaly detection with custom parameters
    main(
        input_csv_path=input_file,
        output_csv_path=output_file,
        method="isolation_forest",
        contamination=0.05,  # Lower contamination for more sensitive detection
        n_estimators=200     # More estimators for better accuracy
    )
    
    # Display results
    results = pd.read_csv(output_file)
    print(f"\nResults summary:")
    print(f"Total rows: {len(results)}")
    print(f"Mean abnormality score: {results['Abnormality_score'].mean():.2f}")
    print(f"Max abnormality score: {results['Abnormality_score'].max():.2f}")


def run_programmatic_example():
    """Run example using the programmatic API."""
    print("\n" + "="*60)
    print("PROGRAMMATIC API EXAMPLE")
    print("="*60)
    
    # Create sample data
    input_file = create_sample_data("sample_programmatic.csv", 200)
    
    # Use individual components
    processor = DataProcessor()
    training_data, analysis_data = processor.process_data(input_file)
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Analysis data shape: {analysis_data.shape}")
    
    # Train model
    detector = AnomalyDetector(method="isolation_forest", contamination=0.1)
    detector.fit(training_data)
    
    # Detect anomalies
    raw_scores = detector.predict(analysis_data)
    
    # Calculate feature contributions
    attributor = FeatureAttributor()
    
    # Get training period scores for normalization
    timestamp_col = training_data.columns[0]
    training_start = datetime(2004, 1, 1, 0, 0)
    training_end = datetime(2004, 1, 5, 23, 59)
    training_mask = (
        (analysis_data[timestamp_col] >= training_start) & 
        (analysis_data[timestamp_col] <= training_end)
    )
    training_scores = raw_scores[training_mask]
    
    abnormality_scores = attributor.calculate_abnormality_score(raw_scores, training_scores)
    feature_importance = detector.get_feature_importance(analysis_data)
    feature_contributions = attributor.calculate_feature_contributions(
        abnormality_scores, feature_importance, analysis_data
    )
    
    # Create output
    output_df = analysis_data.copy()
    output_df['Abnormality_score'] = abnormality_scores
    for col in feature_contributions.columns:
        output_df[col] = feature_contributions[col]
    
    output_file = "output_programmatic.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"\nResults summary:")
    print(f"Total rows: {len(output_df)}")
    print(f"Mean abnormality score: {abnormality_scores.mean():.2f}")
    print(f"Max abnormality score: {abnormality_scores.max():.2f}")
    
    # Show feature importance summary
    print(f"\nFeature importance summary:")
    for feature, importance_scores in feature_importance.items():
        mean_importance = np.mean(importance_scores)
        print(f"  {feature}: {mean_importance:.4f}")


def cleanup_files():
    """Clean up generated files."""
    files_to_remove = [
        "sample_basic.csv", "output_basic.csv",
        "sample_autoencoder.csv", "output_autoencoder.csv",
        "sample_ensemble.csv", "output_ensemble.csv",
        "sample_custom.csv", "output_custom.csv",
        "sample_programmatic.csv", "output_programmatic.csv"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")


def main_example():
    """Run all examples."""
    print("MULTIVARIATE TIME SERIES ANOMALY DETECTION - EXAMPLES")
    print("="*60)
    
    try:
        # Run different examples
        run_basic_example()
        run_autoencoder_example()
        run_ensemble_example()
        run_custom_parameters_example()
        run_programmatic_example()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Ask if user wants to clean up files
        response = input("\nDo you want to clean up the generated files? (y/n): ")
        if response.lower() in ['y', 'yes']:
            cleanup_files()
            print("Cleanup completed.")
        else:
            print("Files preserved for inspection.")
            
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main_example()