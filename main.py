"""
Main Module for Multivariate Time Series Anomaly Detection

This module orchestrates the complete anomaly detection pipeline:
1. Data preprocessing and validation
2. Model training on normal period
3. Anomaly detection on analysis period
4. Feature attribution calculation
5. Output generation with required columns
"""

import pandas as pd
import numpy as np
import warnings
import time
from typing import Optional
from datetime import datetime

from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector
from feature_attributor import FeatureAttributor


def main(input_csv_path: str, output_csv_path: str, 
         method: str = 'isolation_forest', **kwargs) -> None:
    """
    Main function for multivariate time series anomaly detection.
    
    This function implements the complete pipeline according to the requirements:
    - Loads and preprocesses CSV data
    - Trains anomaly detection model on normal period
    - Detects anomalies in analysis period
    - Calculates feature contributions
    - Generates output with required columns
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file
        method: Anomaly detection method ('isolation_forest', 'autoencoder', 'pca', 'ensemble')
        **kwargs: Additional parameters for the anomaly detector
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data processing fails
        RuntimeError: If model training or prediction fails
    """
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("MULTIVARIATE TIME SERIES ANOMALY DETECTION")
        print("=" * 60)
        print(f"Input file: {input_csv_path}")
        print(f"Output file: {output_csv_path}")
        print(f"Method: {method}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Data Processing
        print("Step 1: Data Processing")
        print("-" * 30)
        data_processor = DataProcessor()
        training_data, analysis_data = data_processor.process_data(input_csv_path)
        
        # Create training mask for validation
        timestamp_col = training_data.columns[0]
        training_start = datetime(2004, 1, 1, 0, 0)
        training_end = datetime(2004, 1, 5, 23, 59)
        
        training_mask = (
            (analysis_data[timestamp_col] >= training_start) & 
            (analysis_data[timestamp_col] <= training_end)
        )
        
        print(f"✓ Data processing completed successfully")
        print()
        
        # Step 2: Model Training
        print("Step 2: Model Training")
        print("-" * 30)
        detector = AnomalyDetector(method=method, **kwargs)
        detector.fit(training_data)
        print(f"✓ Model training completed successfully")
        print()
        
        # Step 3: Anomaly Detection
        print("Step 3: Anomaly Detection")
        print("-" * 30)
        raw_scores = detector.predict(analysis_data)
        print(f"✓ Anomaly detection completed successfully")
        print()
        
        # Step 4: Score Calculation and Feature Attribution
        print("Step 4: Score Calculation and Feature Attribution")
        print("-" * 30)
        
        # Get training period scores for normalization
        training_scores = raw_scores[training_mask]
        
        # Calculate abnormality scores
        attributor = FeatureAttributor()
        abnormality_scores = attributor.calculate_abnormality_score(raw_scores, training_scores)
        
        # Validate training period scores
        attributor.validate_training_period_scores(abnormality_scores, training_mask)
        
        # Get feature importance
        feature_importance = detector.get_feature_importance(analysis_data)
        
        # Calculate feature contributions
        feature_contributions = attributor.calculate_feature_contributions(
            abnormality_scores, feature_importance, analysis_data
        )
        
        print(f"✓ Score calculation and feature attribution completed successfully")
        print()
        
        # Step 5: Output Generation
        print("Step 5: Output Generation")
        print("-" * 30)
        
        # Create output DataFrame
        output_df = analysis_data.copy()
        output_df['Abnormality_score'] = abnormality_scores
        
        # Add feature contribution columns
        for col in feature_contributions.columns:
            output_df[col] = feature_contributions[col]
        
        # Save to CSV
        output_df.to_csv(output_csv_path, index=False)
        
        print(f"✓ Output saved to: {output_csv_path}")
        print()
        
        # Step 6: Validation and Summary
        print("Step 6: Validation and Summary")
        print("-" * 30)
        
        # Validate output requirements
        validate_output_requirements(output_df, training_mask)
        
        # Generate summary statistics
        generate_summary_statistics(output_df, training_mask, feature_importance)
        
        # Performance analysis
        analysis_results = attributor.analyze_anomaly_patterns(
            abnormality_scores, analysis_data
        )
        
        print(f"✓ Validation and summary completed successfully")
        print()
        
        # Final summary
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Input rows: {len(analysis_data)}")
        print(f"Output rows: {len(output_df)}")
        print(f"Features analyzed: {len(analysis_data.columns) - 1}")
        print(f"Training period rows: {np.sum(training_mask)}")
        print(f"Analysis period rows: {len(analysis_data)}")
        print(f"Mean abnormality score: {np.mean(abnormality_scores):.2f}")
        print(f"Max abnormality score: {np.max(abnormality_scores):.2f}")
        print(f"Training period mean score: {np.mean(abnormality_scores[training_mask]):.2f}")
        print(f"Training period max score: {np.max(abnormality_scores[training_mask]):.2f}")
        print(f"Sudden jumps detected: {analysis_results['sudden_jumps']}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        raise
    except ValueError as e:
        print(f"❌ Data processing error: {e}")
        raise
    except RuntimeError as e:
        print(f"❌ Runtime error: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


def validate_output_requirements(output_df: pd.DataFrame, training_mask: np.ndarray) -> None:
    """
    Validate that output meets all requirements.
    
    Args:
        output_df: Output DataFrame
        training_mask: Boolean mask for training period
    """
    print("Validating output requirements...")
    
    # Check required columns
    required_columns = ['Abnormality_score', 'top_feature_1', 'top_feature_2', 
                       'top_feature_3', 'top_feature_4', 'top_feature_5', 
                       'top_feature_6', 'top_feature_7']
    
    missing_columns = [col for col in required_columns if col not in output_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check abnormality score range
    abnormality_scores = output_df['Abnormality_score']
    if np.any(abnormality_scores < 0) or np.any(abnormality_scores > 100):
        warnings.warn("Abnormality scores outside 0-100 range detected")
    
    # Check training period scores
    training_scores = abnormality_scores[training_mask]
    if len(training_scores) > 0:
        training_mean = np.mean(training_scores)
        training_max = np.max(training_scores)
        
        if training_mean >= 10:
            warnings.warn(f"Training period mean score ({training_mean:.2f}) >= 10")
        
        if training_max >= 25:
            warnings.warn(f"Training period max score ({training_max:.2f}) >= 25")
    
    print("✓ Output validation passed")


def generate_summary_statistics(output_df: pd.DataFrame, 
                              training_mask: np.ndarray,
                              feature_importance: dict) -> None:
    """
    Generate summary statistics for the analysis.
    
    Args:
        output_df: Output DataFrame
        training_mask: Boolean mask for training period
        feature_importance: Feature importance dictionary
    """
    print("Generating summary statistics...")
    
    abnormality_scores = output_df['Abnormality_score']
    
    # Score distribution
    print(f"  Score distribution:")
    print(f"    Mean: {np.mean(abnormality_scores):.2f}")
    print(f"    Std: {np.std(abnormality_scores):.2f}")
    print(f"    Min: {np.min(abnormality_scores):.2f}")
    print(f"    Max: {np.max(abnormality_scores):.2f}")
    
    # Score categories
    normal_count = np.sum(abnormality_scores <= 10)
    slight_count = np.sum((abnormality_scores > 10) & (abnormality_scores <= 30))
    moderate_count = np.sum((abnormality_scores > 30) & (abnormality_scores <= 60))
    significant_count = np.sum((abnormality_scores > 60) & (abnormality_scores <= 90))
    severe_count = np.sum(abnormality_scores > 90)
    
    print(f"  Anomaly categories:")
    print(f"    Normal (0-10): {normal_count} ({normal_count/len(abnormality_scores)*100:.1f}%)")
    print(f"    Slight (11-30): {slight_count} ({slight_count/len(abnormality_scores)*100:.1f}%)")
    print(f"    Moderate (31-60): {moderate_count} ({moderate_count/len(abnormality_scores)*100:.1f}%)")
    print(f"    Significant (61-90): {significant_count} ({significant_count/len(abnormality_scores)*100:.1f}%)")
    print(f"    Severe (91-100): {severe_count} ({severe_count/len(abnormality_scores)*100:.1f}%)")
    
    # Training period statistics
    training_scores = abnormality_scores[training_mask]
    if len(training_scores) > 0:
        print(f"  Training period statistics:")
        print(f"    Mean: {np.mean(training_scores):.2f}")
        print(f"    Max: {np.max(training_scores):.2f}")
        print(f"    Rows: {len(training_scores)}")
    
    # Feature importance summary
    if feature_importance:
        print(f"  Feature importance summary:")
        for feature, importance_scores in feature_importance.items():
            mean_importance = np.mean(importance_scores)
            print(f"    {feature}: {mean_importance:.4f}")
    
    print("✓ Summary statistics generated")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multivariate Time Series Anomaly Detection"
    )
    parser.add_argument(
        "input_csv_path", 
        help="Path to input CSV file"
    )
    parser.add_argument(
        "output_csv_path", 
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--method", 
        default="isolation_forest",
        choices=["isolation_forest", "autoencoder", "pca", "ensemble"],
        help="Anomaly detection method (default: isolation_forest)"
    )
    parser.add_argument(
        "--contamination", 
        type=float, 
        default=0.1,
        help="Contamination factor for Isolation Forest (default: 0.1)"
    )
    parser.add_argument(
        "--n_estimators", 
        type=int, 
        default=100,
        help="Number of estimators for Isolation Forest (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Run main function
    main(
        input_csv_path=args.input_csv_path,
        output_csv_path=args.output_csv_path,
        method=args.method,
        contamination=args.contamination,
        n_estimators=args.n_estimators
    )