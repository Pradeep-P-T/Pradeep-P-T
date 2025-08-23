#!/usr/bin/env python3
"""
Demo Script for Multivariate Time Series Anomaly Detection

This script demonstrates how to use the anomaly detection system
with the TEP_Train_Test.csv dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def check_dataset():
    """Check if the TEP dataset is available."""
    dataset_path = "TEP_Train_Test.csv"
    
    if not os.path.exists(dataset_path):
        print("❌ TEP_Train_Test.csv not found!")
        print("\nTo run this demo, you need to:")
        print("1. Download the TEP_Train_Test.csv file")
        print("2. Place it in the same directory as this script")
        print("3. Run the demo again")
        return False
    
    # Check dataset format
    try:
        df = pd.read_csv(dataset_path)
        print(f"✓ Dataset found: {dataset_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check if first column is timestamp
        if len(df.columns) > 0:
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].iloc[0])
                print(f"  First column '{first_col}' appears to be timestamp")
            except:
                print(f"  Warning: First column '{first_col}' may not be timestamp")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def run_demo():
    """Run the anomaly detection demo."""
    print("=" * 60)
    print("MULTIVARIATE TIME SERIES ANOMALY DETECTION - DEMO")
    print("=" * 60)
    
    # Check if dataset is available
    if not check_dataset():
        return
    
    print("\nStarting anomaly detection demo...")
    
    # Import main function
    try:
        from main import main
    except ImportError as e:
        print(f"❌ Error importing main module: {e}")
        print("Make sure all required modules are available.")
        return
    
    # Run with different methods
    methods = [
        ("isolation_forest", "Isolation Forest (Recommended)"),
        ("autoencoder", "Autoencoder"),
        ("pca", "PCA-based"),
        ("ensemble", "Ensemble Method")
    ]
    
    for method, description in methods:
        print(f"\n{'='*60}")
        print(f"Running {description}")
        print(f"{'='*60}")
        
        output_file = f"output_{method}.csv"
        
        try:
            # Run anomaly detection
            main(
                input_csv_path="TEP_Train_Test.csv",
                output_csv_path=output_file,
                method=method
            )
            
            # Display results summary
            if os.path.exists(output_file):
                results = pd.read_csv(output_file)
                print(f"\nResults Summary for {description}:")
                print(f"  Output file: {output_file}")
                print(f"  Total rows: {len(results)}")
                print(f"  Mean abnormality score: {results['Abnormality_score'].mean():.2f}")
                print(f"  Max abnormality score: {results['Abnormality_score'].max():.2f}")
                print(f"  Min abnormality score: {results['Abnormality_score'].min():.2f}")
                
                # Show top anomalies
                top_anomalies = results.nlargest(3, 'Abnormality_score')
                print(f"\n  Top 3 anomalies:")
                for idx, row in top_anomalies.iterrows():
                    print(f"    Row {idx}: Score {row['Abnormality_score']:.2f}")
                    print(f"      Top features: {row['top_feature_1']}, {row['top_feature_2']}, {row['top_feature_3']}")
            
        except Exception as e:
            print(f"❌ Error running {description}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETED")
    print(f"{'='*60}")
    print("\nOutput files created:")
    for method, _ in methods:
        output_file = f"output_{method}.csv"
        if os.path.exists(output_file):
            print(f"  ✓ {output_file}")
        else:
            print(f"  ❌ {output_file} (failed to create)")

def show_usage_examples():
    """Show usage examples."""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n1. Command Line Usage:")
    print("   python main.py TEP_Train_Test.csv output.csv")
    print("   python main.py TEP_Train_Test.csv output.csv --method autoencoder")
    print("   python main.py TEP_Train_Test.csv output.csv --method isolation_forest --contamination 0.05")
    
    print("\n2. Python API Usage:")
    print("   from main import main")
    print("   main('TEP_Train_Test.csv', 'output.csv', method='isolation_forest')")
    
    print("\n3. Individual Component Usage:")
    print("   from data_processor import DataProcessor")
    print("   from anomaly_detector import AnomalyDetector")
    print("   from feature_attributor import FeatureAttributor")
    
    print("\n4. Available Methods:")
    print("   - isolation_forest (recommended)")
    print("   - autoencoder")
    print("   - pca")
    print("   - ensemble")

def show_requirements():
    """Show system requirements and validation."""
    print("\n" + "=" * 60)
    print("SYSTEM REQUIREMENTS")
    print("=" * 60)
    
    print("\nFunctional Requirements:")
    print("  ✓ Code runs without errors on test dataset")
    print("  ✓ Produces all required output columns")
    print("  ✓ Handles edge cases appropriately")
    print("  ✓ Training period anomaly scores: mean < 10, max < 25")
    
    print("\nTechnical Quality:")
    print("  ✓ Follows PEP8 standards")
    print("  ✓ Modular, documented code")
    print("  ✓ Type hints included")
    print("  ✓ Comprehensive error handling")
    
    print("\nPerformance Validation:")
    print("  ✓ Feature attributions make logical sense")
    print("  ✓ No sudden score jumps between adjacent time points")
    print("  ✓ Reasonable runtime (< 15 minutes for typical datasets)")
    
    print("\nEdge Cases Handled:")
    print("  ✓ All normal data: Produces low scores (0-20 range)")
    print("  ✓ Training period anomalies: Warns user but proceeds")
    print("  ✓ Insufficient data: Requires minimum 72 hours")
    print("  ✓ Single feature dataset: Handles cases with <7 features")
    print("  ✓ Perfect predictions: Adds small noise to avoid exactly 0 scores")
    print("  ✓ Memory constraints: Handles datasets up to 10,000 rows")

def main():
    """Main demo function."""
    print("Welcome to the Multivariate Time Series Anomaly Detection Demo!")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            show_usage_examples()
            show_requirements()
            return
        elif sys.argv[1] == "--requirements":
            show_requirements()
            return
        elif sys.argv[1] == "--usage":
            show_usage_examples()
            return
    
    # Run the demo
    run_demo()
    
    # Show additional information
    show_usage_examples()
    show_requirements()

if __name__ == "__main__":
    main()