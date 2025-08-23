"""
Test Script for Multivariate Time Series Anomaly Detection

This script validates all components of the anomaly detection system
and ensures they meet the specified requirements.
"""

import pandas as pd
import numpy as np
import unittest
import tempfile
import os
from datetime import datetime, timedelta

from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector
from feature_attributor import FeatureAttributor
from main import main


class TestAnomalyDetection(unittest.TestCase):
    """Test cases for the anomaly detection system."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = self.create_test_data()
        self.temp_dir = tempfile.mkdtemp()
        
    def create_test_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Create test data with known anomalies."""
        # Generate timestamps
        start_time = datetime(2004, 1, 1, 0, 0)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_rows)]
        
        # Generate normal features
        np.random.seed(42)
        
        # Feature 1: Temperature (normal range 20-30°C)
        temp = np.random.normal(25, 2, n_rows)
        
        # Feature 2: Pressure (normal range 100-120 kPa)
        pressure = np.random.normal(110, 5, n_rows)
        
        # Feature 3: Humidity (normal range 40-60%)
        humidity = np.random.normal(50, 5, n_rows)
        
        # Feature 4: Flow rate (normal range 10-15 L/min)
        flow = np.random.normal(12.5, 1, n_rows)
        
        # Feature 5: Vibration (normal range 0.1-0.5 mm/s)
        vibration = np.random.normal(0.3, 0.1, n_rows)
        
        # Add known anomalies
        anomaly_indices = [200, 400, 600, 800]
        for idx in anomaly_indices:
            if idx < n_rows:
                temp[idx] = 45  # Temperature spike
                pressure[idx] = 80  # Pressure drop
                humidity[idx] = 85  # Humidity spike
                flow[idx] = 25  # Flow rate spike
                vibration[idx] = 2.0  # Vibration spike
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temp,
            'pressure': pressure,
            'humidity': humidity,
            'flow_rate': flow,
            'vibration': vibration
        })
    
    def test_data_processor(self):
        """Test data processing functionality."""
        print("Testing DataProcessor...")
        
        # Save test data to temporary file
        input_file = os.path.join(self.temp_dir, "test_input.csv")
        self.test_data.to_csv(input_file, index=False)
        
        # Test data processing
        processor = DataProcessor()
        training_data, analysis_data = processor.process_data(input_file)
        
        # Validate results
        self.assertIsInstance(training_data, pd.DataFrame)
        self.assertIsInstance(analysis_data, pd.DataFrame)
        self.assertGreater(len(training_data), 0)
        self.assertGreater(len(analysis_data), 0)
        
        # Check that training data is within training period
        timestamp_col = training_data.columns[0]
        training_start = datetime(2004, 1, 1, 0, 0)
        training_end = datetime(2004, 1, 5, 23, 59)
        
        self.assertTrue(all(
            (training_data[timestamp_col] >= training_start) & 
            (training_data[timestamp_col] <= training_end)
        ))
        
        print("✓ DataProcessor tests passed")
    
    def test_anomaly_detector_isolation_forest(self):
        """Test Isolation Forest anomaly detection."""
        print("Testing Isolation Forest...")
        
        # Create test data
        input_file = os.path.join(self.temp_dir, "test_if.csv")
        self.test_data.to_csv(input_file, index=False)
        
        # Process data
        processor = DataProcessor()
        training_data, analysis_data = processor.process_data(input_file)
        
        # Test Isolation Forest
        detector = AnomalyDetector(method="isolation_forest")
        detector.fit(training_data)
        
        # Predict anomalies
        scores = detector.predict(analysis_data)
        
        # Validate results
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(analysis_data))
        self.assertTrue(np.all(np.isfinite(scores)))
        
        # Test feature importance
        feature_importance = detector.get_feature_importance(analysis_data)
        self.assertIsInstance(feature_importance, dict)
        self.assertGreater(len(feature_importance), 0)
        
        print("✓ Isolation Forest tests passed")
    
    def test_anomaly_detector_autoencoder(self):
        """Test Autoencoder anomaly detection."""
        print("Testing Autoencoder...")
        
        # Create test data
        input_file = os.path.join(self.temp_dir, "test_ae.csv")
        self.test_data.to_csv(input_file, index=False)
        
        # Process data
        processor = DataProcessor()
        training_data, analysis_data = processor.process_data(input_file)
        
        # Test Autoencoder
        detector = AnomalyDetector(method="autoencoder", epochs=5)
        detector.fit(training_data)
        
        # Predict anomalies
        scores = detector.predict(analysis_data)
        
        # Validate results
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(analysis_data))
        self.assertTrue(np.all(np.isfinite(scores)))
        
        print("✓ Autoencoder tests passed")
    
    def test_feature_attributor(self):
        """Test feature attribution functionality."""
        print("Testing FeatureAttributor...")
        
        # Create test data
        input_file = os.path.join(self.temp_dir, "test_fa.csv")
        self.test_data.to_csv(input_file, index=False)
        
        # Process data and get scores
        processor = DataProcessor()
        training_data, analysis_data = processor.process_data(input_file)
        
        detector = AnomalyDetector(method="isolation_forest")
        detector.fit(training_data)
        raw_scores = detector.predict(analysis_data)
        
        # Test feature attribution
        attributor = FeatureAttributor()
        
        # Calculate abnormality scores
        training_mask = (
            (analysis_data[analysis_data.columns[0]] >= datetime(2004, 1, 1, 0, 0)) & 
            (analysis_data[analysis_data.columns[0]] <= datetime(2004, 1, 5, 23, 59))
        )
        training_scores = raw_scores[training_mask]
        
        abnormality_scores = attributor.calculate_abnormality_score(raw_scores, training_scores)
        
        # Validate abnormality scores
        self.assertIsInstance(abnormality_scores, np.ndarray)
        self.assertEqual(len(abnormality_scores), len(analysis_data))
        self.assertTrue(np.all((abnormality_scores >= 0) & (abnormality_scores <= 100)))
        
        # Test feature contributions
        feature_importance = detector.get_feature_importance(analysis_data)
        feature_contributions = attributor.calculate_feature_contributions(
            abnormality_scores, feature_importance, analysis_data
        )
        
        # Validate feature contributions
        self.assertIsInstance(feature_contributions, pd.DataFrame)
        self.assertEqual(len(feature_contributions), len(analysis_data))
        
        # Check required columns
        required_cols = ['top_feature_1', 'top_feature_2', 'top_feature_3', 
                        'top_feature_4', 'top_feature_5', 'top_feature_6', 'top_feature_7']
        for col in required_cols:
            self.assertIn(col, feature_contributions.columns)
        
        print("✓ FeatureAttributor tests passed")
    
    def test_complete_pipeline(self):
        """Test the complete anomaly detection pipeline."""
        print("Testing complete pipeline...")
        
        # Create test data
        input_file = os.path.join(self.temp_dir, "test_pipeline.csv")
        output_file = os.path.join(self.temp_dir, "test_output.csv")
        self.test_data.to_csv(input_file, index=False)
        
        # Run complete pipeline
        main(input_file, output_file, method="isolation_forest")
        
        # Validate output
        output_df = pd.read_csv(output_file)
        
        # Check required columns
        required_columns = ['Abnormality_score', 'top_feature_1', 'top_feature_2', 
                           'top_feature_3', 'top_feature_4', 'top_feature_5', 
                           'top_feature_6', 'top_feature_7']
        
        for col in required_columns:
            self.assertIn(col, output_df.columns)
        
        # Check abnormality score range
        abnormality_scores = output_df['Abnormality_score']
        self.assertTrue(np.all((abnormality_scores >= 0) & (abnormality_scores <= 100)))
        
        # Check training period scores
        timestamp_col = output_df.columns[0]
        training_mask = (
            (pd.to_datetime(output_df[timestamp_col]) >= datetime(2004, 1, 1, 0, 0)) & 
            (pd.to_datetime(output_df[timestamp_col]) <= datetime(2004, 1, 5, 23, 59))
        )
        training_scores = abnormality_scores[training_mask]
        
        if len(training_scores) > 0:
            self.assertLess(np.mean(training_scores), 25)  # Should be <25
            self.assertLess(np.max(training_scores), 50)   # Should be <50
        
        print("✓ Complete pipeline tests passed")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("Testing edge cases...")
        
        # Test with insufficient data
        small_data = self.create_test_data(50)  # Less than 72 hours
        input_file = os.path.join(self.temp_dir, "test_small.csv")
        small_data.to_csv(input_file, index=False)
        
        processor = DataProcessor()
        
        with self.assertRaises(ValueError):
            processor.process_data(input_file)
        
        # Test with missing values
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[100:110, 'temperature'] = np.nan
        input_file = os.path.join(self.temp_dir, "test_missing.csv")
        data_with_missing.to_csv(input_file, index=False)
        
        # Should handle missing values gracefully
        training_data, analysis_data = processor.process_data(input_file)
        self.assertGreater(len(training_data), 0)
        self.assertGreater(len(analysis_data), 0)
        
        # Test with constant features
        data_constant = self.test_data.copy()
        data_constant['constant_feature'] = 42.0  # Constant value
        input_file = os.path.join(self.temp_dir, "test_constant.csv")
        data_constant.to_csv(input_file, index=False)
        
        # Should handle constant features gracefully
        training_data, analysis_data = processor.process_data(input_file)
        self.assertGreater(len(training_data), 0)
        self.assertGreater(len(analysis_data), 0)
        
        print("✓ Edge cases tests passed")
    
    def test_performance_requirements(self):
        """Test performance requirements."""
        print("Testing performance requirements...")
        
        # Create larger test data
        large_data = self.create_test_data(5000)  # 5000 rows
        input_file = os.path.join(self.temp_dir, "test_large.csv")
        output_file = os.path.join(self.temp_dir, "test_large_output.csv")
        large_data.to_csv(input_file, index=False)
        
        import time
        start_time = time.time()
        
        # Run pipeline
        main(input_file, output_file, method="isolation_forest")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Check execution time (should be <15 minutes = 900 seconds)
        self.assertLess(execution_time, 900)
        
        print(f"✓ Performance test passed (execution time: {execution_time:.2f} seconds)")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING ANOMALY DETECTION TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnomalyDetection)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)