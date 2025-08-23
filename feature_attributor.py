"""
Feature Attribution Module for Anomaly Detection

This module calculates and ranks feature contributions for each anomaly,
following the specified requirements for top contributor identification.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from scipy import stats


class FeatureAttributor:
    """
    Calculates and ranks feature contributions for anomaly detection.
    
    This class implements the feature attribution logic according to the requirements:
    - Rank features by absolute contribution magnitude
    - Break ties alphabetically
    - Apply 1% threshold filter
    - Select top 7 contributing features
    - Fill remaining slots with empty strings if <7 features
    """
    
    def __init__(self, min_contribution_threshold: float = 0.01):
        """
        Initialize the feature attributor.
        
        Args:
            min_contribution_threshold: Minimum contribution threshold (default: 1%)
        """
        self.min_contribution_threshold = min_contribution_threshold
        
    def calculate_feature_contributions(self, 
                                      anomaly_scores: np.ndarray,
                                      feature_importance: Dict[str, np.ndarray],
                                      data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate feature contributions for each row.
        
        Args:
            anomaly_scores: Array of anomaly scores for each row
            feature_importance: Dictionary with feature importance scores
            data: Original DataFrame with features
            
        Returns:
            DataFrame with feature contribution rankings
        """
        n_rows = len(data)
        feature_names = list(feature_importance.keys())
        
        # Initialize results
        top_features = []
        
        for row_idx in range(n_rows):
            # Calculate contributions for this row
            row_contributions = self._calculate_row_contributions(
                row_idx, anomaly_scores, feature_importance, feature_names
            )
            
            # Get top 7 features for this row
            top_7_features = self._get_top_features(row_contributions)
            top_features.append(top_7_features)
        
        # Create DataFrame with top feature columns
        result_df = pd.DataFrame(top_features, columns=[
            'top_feature_1', 'top_feature_2', 'top_feature_3', 'top_feature_4',
            'top_feature_5', 'top_feature_6', 'top_feature_7'
        ])
        
        return result_df
    
    def _calculate_row_contributions(self, 
                                   row_idx: int,
                                   anomaly_scores: np.ndarray,
                                   feature_importance: Dict[str, np.ndarray],
                                   feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate feature contributions for a specific row.
        
        Args:
            row_idx: Index of the row to analyze
            anomaly_scores: Array of anomaly scores
            feature_importance: Dictionary with feature importance scores
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature names and their contribution scores
        """
        row_score = anomaly_scores[row_idx]
        
        # If anomaly score is very low, all features contribute equally
        if row_score < 1e-6:
            return {feature: 0.0 for feature in feature_names}
        
        contributions = {}
        total_contribution = 0.0
        
        for feature in feature_names:
            # Get feature importance for this row
            if feature in feature_importance:
                importance = feature_importance[feature][row_idx]
            else:
                importance = 0.0
            
            # Calculate contribution as a percentage of total anomaly score
            contribution = abs(importance) / max(row_score, 1e-6)
            contributions[feature] = contribution
            total_contribution += contribution
        
        # Normalize contributions to sum to 1.0
        if total_contribution > 0:
            for feature in contributions:
                contributions[feature] /= total_contribution
        
        return contributions
    
    def _get_top_features(self, contributions: Dict[str, float]) -> List[str]:
        """
        Get top 7 contributing features for a row.
        
        Args:
            contributions: Dictionary with feature contributions
            
        Returns:
            List of top 7 feature names (or empty strings if <7)
        """
        # Filter features by minimum contribution threshold
        filtered_contributions = {
            feature: score for feature, score in contributions.items()
            if score >= self.min_contribution_threshold
        }
        
        if not filtered_contributions:
            # If no features meet threshold, return empty strings
            return [''] * 7
        
        # Sort features by contribution magnitude (descending), then alphabetically
        sorted_features = sorted(
            filtered_contributions.items(),
            key=lambda x: (-x[1], x[0])  # Sort by contribution (desc), then alphabetically
        )
        
        # Extract feature names
        top_features = [feature for feature, _ in sorted_features]
        
        # Take top 7 or pad with empty strings
        if len(top_features) >= 7:
            return top_features[:7]
        else:
            return top_features + [''] * (7 - len(top_features))
    
    def calculate_abnormality_score(self, 
                                  raw_scores: np.ndarray,
                                  training_scores: np.ndarray) -> np.ndarray:
        """
        Transform raw anomaly scores to 0-100 scale using percentile ranking.
        
        Args:
            raw_scores: Raw anomaly scores for all data
            training_scores: Raw anomaly scores for training period only
            
        Returns:
            Array of abnormality scores (0-100)
        """
        # Calculate percentiles based on training data distribution
        if len(training_scores) > 0:
            # Use training data percentiles for normalization
            percentiles = np.percentile(training_scores, np.arange(0, 101, 1))
            
            # Map raw scores to percentiles
            abnormality_scores = np.zeros_like(raw_scores)
            for i, score in enumerate(raw_scores):
                # Find percentile rank
                percentile_rank = np.searchsorted(percentiles, score)
                abnormality_scores[i] = min(percentile_rank, 100)
        else:
            # Fallback: use min-max normalization
            min_score = np.min(raw_scores)
            max_score = np.max(raw_scores)
            if max_score > min_score:
                abnormality_scores = ((raw_scores - min_score) / (max_score - min_score)) * 100
            else:
                abnormality_scores = np.zeros_like(raw_scores)
        
        # Ensure scores are within 0-100 range
        abnormality_scores = np.clip(abnormality_scores, 0, 100)
        
        # Add small noise to avoid exactly 0 scores (as per requirements)
        noise = np.random.normal(0, 0.01, len(abnormality_scores))
        abnormality_scores += noise
        abnormality_scores = np.clip(abnormality_scores, 0, 100)
        
        return abnormality_scores
    
    def validate_training_period_scores(self, 
                                      abnormality_scores: np.ndarray,
                                      training_mask: np.ndarray) -> None:
        """
        Validate that training period scores meet requirements.
        
        Args:
            abnormality_scores: Array of abnormality scores
            training_mask: Boolean mask indicating training period rows
            
        Raises:
            Warning: If training period scores don't meet requirements
        """
        training_scores = abnormality_scores[training_mask]
        
        if len(training_scores) > 0:
            mean_score = np.mean(training_scores)
            max_score = np.max(training_scores)
            
            if mean_score >= 10:
                import warnings
                warnings.warn(f"Training period mean score ({mean_score:.2f}) >= 10")
            
            if max_score >= 25:
                import warnings
                warnings.warn(f"Training period max score ({max_score:.2f}) >= 25")
    
    def get_feature_contribution_summary(self, 
                                       feature_importance: Dict[str, np.ndarray],
                                       data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of feature contributions across the dataset.
        
        Args:
            feature_importance: Dictionary with feature importance scores
            data: Original DataFrame
            
        Returns:
            DataFrame with feature contribution statistics
        """
        feature_names = list(feature_importance.keys())
        summary_data = []
        
        for feature in feature_names:
            importance_scores = feature_importance[feature]
            
            summary_data.append({
                'feature': feature,
                'mean_importance': np.mean(importance_scores),
                'std_importance': np.std(importance_scores),
                'max_importance': np.max(importance_scores),
                'min_importance': np.min(importance_scores),
                'contribution_rate': np.mean(importance_scores > 0)
            })
        
        return pd.DataFrame(summary_data)
    
    def analyze_anomaly_patterns(self, 
                               abnormality_scores: np.ndarray,
                               data: pd.DataFrame,
                               window_size: int = 5) -> Dict[str, Any]:
        """
        Analyze patterns in anomaly scores to detect sudden jumps.
        
        Args:
            abnormality_scores: Array of abnormality scores
            data: Original DataFrame
            window_size: Size of sliding window for analysis
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate score differences between adjacent time points
        score_diffs = np.abs(np.diff(abnormality_scores))
        
        # Find sudden jumps (large differences)
        jump_threshold = np.percentile(score_diffs, 95)  # 95th percentile
        sudden_jumps = score_diffs > jump_threshold
        
        # Calculate statistics
        analysis_results = {
            'total_rows': len(abnormality_scores),
            'sudden_jumps': np.sum(sudden_jumps),
            'jump_rate': np.mean(sudden_jumps),
            'max_score_diff': np.max(score_diffs),
            'mean_score_diff': np.mean(score_diffs),
            'score_range': np.max(abnormality_scores) - np.min(abnormality_scores),
            'score_variance': np.var(abnormality_scores)
        }
        
        return analysis_results