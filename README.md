# Multivariate Time Series Anomaly Detection

A comprehensive Python-based machine learning solution for detecting anomalies in multivariate time series data and identifying the primary contributing features for each anomaly.

## Overview

This project implements a complete anomaly detection pipeline that:

- **Detects anomalies** in multivariate time series data using multiple ML techniques
- **Identifies contributing features** for each anomaly (top 7 features)
- **Generates abnormality scores** on a 0-100 scale
- **Handles data quality issues** including missing values and edge cases
- **Follows modular design** with comprehensive error handling

## Features

### Supported Anomaly Detection Methods
- **Isolation Forest** (recommended) - Built-in feature importance capabilities
- **Autoencoders** - For complex pattern detection
- **PCA-based** - For dimensionality reduction and reconstruction error
- **Ensemble Methods** - Combining multiple techniques for robust results

### Key Capabilities
- **Data Preprocessing**: Handles missing values, invalid data, timestamp validation
- **Feature Attribution**: Identifies top 7 contributing features for each anomaly
- **Score Normalization**: Transforms raw scores to 0-100 scale using percentile ranking
- **Edge Case Handling**: Manages insufficient data, constant features, memory constraints
- **Validation**: Ensures training period scores meet requirements (<10 mean, <25 max)

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd multivariate-time-series-anomaly-detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python main.py input.csv output.csv --method isolation_forest
```

#### Parameters:
- `input_csv_path`: Path to input CSV file
- `output_csv_path`: Path to output CSV file
- `--method`: Anomaly detection method (`isolation_forest`, `autoencoder`, `pca`, `ensemble`)
- `--contamination`: Contamination factor for Isolation Forest (default: 0.1)
- `--n_estimators`: Number of estimators for Isolation Forest (default: 100)

### Python API

```python
from main import main

# Run anomaly detection
main(
    input_csv_path="TEP_Train_Test.csv",
    output_csv_path="output_with_anomalies.csv",
    method="isolation_forest",
    contamination=0.1,
    n_estimators=100
)
```

## Input Data Format

The input CSV file should have:
- **First column**: Timestamp (datetime format)
- **Remaining columns**: Numerical features for analysis
- **Time period**: Data from 1/1/2004 0:00 to 1/19/2004 7:59

### Example Input:
```csv
timestamp,feature1,feature2,feature3
2004-01-01 00:00:00,23.5,45.2,12.8
2004-01-01 00:01:00,23.7,45.1,12.9
...
```

## Output Format

The output CSV contains all original columns plus 8 new columns:

1. **Abnormality_score**: Float values from 0.0 to 100.0
2. **top_feature_1** through **top_feature_7**: String values containing original column names

### Example Output:
```csv
timestamp,feature1,feature2,feature3,Abnormality_score,top_feature_1,top_feature_2,top_feature_3,top_feature_4,top_feature_5,top_feature_6,top_feature_7
2004-01-01 00:00:00,23.5,45.2,12.8,5.2,feature2,feature1,feature3,,,,
2004-01-01 00:01:00,23.7,45.1,12.9,45.8,feature1,feature3,feature2,,,,
...
```

## Project Structure

```
├── main.py                 # Main orchestration script
├── data_processor.py       # Data loading and preprocessing
├── anomaly_detector.py     # Anomaly detection models
├── feature_attributor.py   # Feature contribution calculation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Technical Details

### Data Processing
- **Training Period**: 1/1/2004 0:00 to 1/5/2004 23:59 (120 hours)
- **Analysis Period**: 1/1/2004 0:00 to 1/19/2004 7:59 (439 hours)
- **Minimum Training Data**: 72 hours required
- **Data Quality**: Handles missing values, invalid data, constant features

### Anomaly Detection
- **Threshold Violation**: Individual variables exceeding normal ranges
- **Relationship Change**: Variables no longer following usual correlations
- **Pattern Deviations**: Temporal sequences differing from normal patterns

### Score Calculation
- **0-10**: Normal behavior (expected for training period)
- **11-30**: Slightly unusual but acceptable
- **31-60**: Moderate anomaly requiring attention
- **61-90**: Significant anomaly needing investigation
- **91-100**: Severe anomaly requiring immediate action

### Feature Attribution
- **Ranking**: By absolute contribution magnitude
- **Tie-breaking**: Alphabetical order of feature names
- **Threshold**: Only features contributing >1% included
- **Padding**: Empty strings for <7 contributing features

## Edge Cases Handled

1. **All normal data**: Produces low scores (0-20 range)
2. **Training period anomalies**: Warns user but proceeds with training
3. **Insufficient data**: Requires minimum 72 hours of training data
4. **Single feature dataset**: Handles cases with <7 features
5. **Perfect predictions**: Adds small noise to avoid exactly 0 scores
6. **Memory constraints**: Handles datasets up to 10,000 rows efficiently

## Performance Requirements

- **Runtime**: <15 minutes for typical datasets
- **Memory**: Efficient handling of up to 10,000 rows
- **Accuracy**: Training period anomaly scores mean <10, max <25
- **Stability**: No sudden score jumps between adjacent time points

## Examples

### Basic Usage
```bash
# Run with default settings (Isolation Forest)
python main.py TEP_Train_Test.csv output.csv

# Run with Autoencoder
python main.py TEP_Train_Test.csv output.csv --method autoencoder

# Run with custom parameters
python main.py TEP_Train_Test.csv output.csv --method isolation_forest --contamination 0.05 --n_estimators 200
```

### Programmatic Usage
```python
from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector
from feature_attributor import FeatureAttributor

# Process data
processor = DataProcessor()
training_data, analysis_data = processor.process_data("input.csv")

# Train model
detector = AnomalyDetector(method="isolation_forest")
detector.fit(training_data)

# Detect anomalies
raw_scores = detector.predict(analysis_data)

# Calculate feature contributions
attributor = FeatureAttributor()
abnormality_scores = attributor.calculate_abnormality_score(raw_scores, raw_scores[:len(training_data)])
feature_importance = detector.get_feature_importance(analysis_data)
feature_contributions = attributor.calculate_feature_contributions(
    abnormality_scores, feature_importance, analysis_data
)
```

## Validation

The system includes comprehensive validation:

- **Output Requirements**: Ensures all required columns are present
- **Score Range**: Validates abnormality scores are 0-100
- **Training Period**: Checks training period scores meet requirements
- **Feature Attribution**: Validates top 7 features are correctly identified
- **Performance**: Monitors execution time and memory usage

## Troubleshooting

### Common Issues

1. **File not found**: Ensure input CSV file exists and path is correct
2. **Insufficient training data**: Verify at least 72 hours of data in training period
3. **Memory errors**: For large datasets, consider using smaller batch sizes
4. **Model convergence**: For autoencoders, try adjusting epochs or learning rate

### Error Messages

- `FileNotFoundError`: Input CSV file doesn't exist
- `ValueError`: Data format issues or insufficient training data
- `RuntimeError`: Model training or prediction failures
- `MemoryError`: Dataset too large for available memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Isolation Forest: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
- Autoencoders: Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks.
- PCA-based anomaly detection: Shyu, M. L., et al. (2003). A novel anomaly detection scheme based on principal component classifier.

