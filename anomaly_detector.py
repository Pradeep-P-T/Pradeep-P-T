"""
Anomaly Detection Module for Multivariate Time Series

This module implements various anomaly detection techniques including
Isolation Forest, Autoencoders, and PCA-based methods.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings


class AnomalyDetector:
    """
    Implements various anomaly detection techniques for multivariate time series.
    
    Supports:
    - Isolation Forest (recommended for built-in feature importance)
    - Autoencoders (for complex patterns)
    - PCA-based methods (for dimensionality reduction)
    - Ensemble methods (combining multiple techniques)
    """
    
    def __init__(self, method: str = 'isolation_forest', **kwargs):
        """
        Initialize the anomaly detector.
        
        Args:
            method: Detection method ('isolation_forest', 'autoencoder', 'pca', 'ensemble')
            **kwargs: Additional parameters for the chosen method
        """
        self.method = method
        self.kwargs = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from DataFrame.
        
        Args:
            df: DataFrame with features (skip timestamp column)
            
        Returns:
            Feature matrix as numpy array
        """
        feature_cols = df.columns[1:]  # Skip timestamp column
        self.feature_names = list(feature_cols)
        return df[feature_cols].values
    
    def _build_autoencoder(self, input_dim: int) -> keras.Model:
        """
        Build autoencoder model for anomaly detection.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled autoencoder model
        """
        # Encoder
        encoder = keras.Sequential([
            layers.Dense(input_dim, activation='relu', input_shape=(input_dim,)),
            layers.Dense(max(input_dim // 2, 8), activation='relu'),
            layers.Dense(max(input_dim // 4, 4), activation='relu')
        ])
        
        # Decoder
        decoder = keras.Sequential([
            layers.Dense(max(input_dim // 4, 4), activation='relu'),
            layers.Dense(max(input_dim // 2, 8), activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        
        # Autoencoder
        autoencoder = keras.Sequential([encoder, decoder])
        
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def fit(self, training_data: pd.DataFrame) -> None:
        """
        Train the anomaly detection model on normal data.
        
        Args:
            training_data: DataFrame with normal period data
        """
        # Prepare features
        X_train = self._prepare_features(training_data)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.method == 'isolation_forest':
            # Isolation Forest
            contamination = self.kwargs.get('contamination', 0.1)
            n_estimators = self.kwargs.get('n_estimators', 100)
            
            self.model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42
            )
            self.model.fit(X_train_scaled)
            
        elif self.method == 'autoencoder':
            # Autoencoder
            epochs = self.kwargs.get('epochs', 50)
            batch_size = self.kwargs.get('batch_size', 32)
            validation_split = self.kwargs.get('validation_split', 0.2)
            
            self.model = self._build_autoencoder(X_train_scaled.shape[1])
            
            # Train autoencoder
            self.model.fit(
                X_train_scaled, X_train_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0
            )
            
        elif self.method == 'pca':
            # PCA-based anomaly detection
            n_components = self.kwargs.get('n_components', None)
            if n_components is None:
                n_components = min(X_train_scaled.shape[1], 10)
            
            self.model = PCA(n_components=n_components)
            self.model.fit(X_train_scaled)
            
        elif self.method == 'ensemble':
            # Ensemble of multiple methods
            self.models = {}
            
            # Isolation Forest
            self.models['isolation_forest'] = IsolationForest(
                contamination=0.1, n_estimators=100, random_state=42
            )
            self.models['isolation_forest'].fit(X_train_scaled)
            
            # PCA
            n_components = min(X_train_scaled.shape[1], 10)
            self.models['pca'] = PCA(n_components=n_components)
            self.models['pca'].fit(X_train_scaled)
            
            # Autoencoder
            self.models['autoencoder'] = self._build_autoencoder(X_train_scaled.shape[1])
            self.models['autoencoder'].fit(
                X_train_scaled, X_train_scaled,
                epochs=30, batch_size=32, validation_split=0.2, verbose=0
            )
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        print(f"Model trained using {self.method} method")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores for the given data.
        
        Args:
            data: DataFrame with data to analyze
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X = self._prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'isolation_forest':
            # Isolation Forest returns -1 for anomalies, 1 for normal
            # Convert to positive scores where higher = more anomalous
            scores = -self.model.decision_function(X_scaled)
            
        elif self.method == 'autoencoder':
            # Autoencoder reconstruction error
            reconstructed = self.model.predict(X_scaled)
            scores = mean_squared_error(X_scaled, reconstructed, multioutput='raw_values')
            # Sum across features for overall anomaly score
            scores = np.sum(scores, axis=1)
            
        elif self.method == 'pca':
            # PCA reconstruction error
            reconstructed = self.model.inverse_transform(self.model.transform(X_scaled))
            scores = np.sum((X_scaled - reconstructed) ** 2, axis=1)
            
        elif self.method == 'ensemble':
            # Combine scores from multiple models
            scores_list = []
            
            # Isolation Forest scores
            if_scores = -self.models['isolation_forest'].decision_function(X_scaled)
            scores_list.append(if_scores)
            
            # PCA scores
            pca_reconstructed = self.models['pca'].inverse_transform(
                self.models['pca'].transform(X_scaled)
            )
            pca_scores = np.sum((X_scaled - pca_reconstructed) ** 2, axis=1)
            scores_list.append(pca_scores)
            
            # Autoencoder scores
            ae_reconstructed = self.models['autoencoder'].predict(X_scaled)
            ae_scores = np.sum((X_scaled - ae_reconstructed) ** 2, axis=1)
            scores_list.append(ae_scores)
            
            # Normalize and combine scores
            scores_array = np.column_stack(scores_list)
            scores_array = (scores_array - scores_array.mean(axis=0)) / scores_array.std(axis=0)
            scores = np.mean(scores_array, axis=1)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return scores
    
    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get feature importance/contribution for anomaly detection.
        
        Args:
            data: DataFrame with data to analyze
            
        Returns:
            Dictionary with feature names as keys and importance scores as values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        X = self._prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        feature_importance = {}
        
        if self.method == 'isolation_forest':
            # Use feature importances from Isolation Forest
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                # Calculate approximate feature importance using permutation
                importances = self._calculate_permutation_importance(X_scaled)
            
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = importances[i] * np.ones(len(data))
                
        elif self.method == 'autoencoder':
            # Calculate gradient-based feature importance
            feature_importance = self._calculate_gradient_importance(X_scaled)
            
        elif self.method == 'pca':
            # Use PCA component loadings
            loadings = self.model.components_
            explained_variance = self.model.explained_variance_ratio_
            
            # Calculate feature importance based on loadings and explained variance
            for i, feature in enumerate(self.feature_names):
                importance = np.sum(np.abs(loadings[:, i]) * explained_variance)
                feature_importance[feature] = importance * np.ones(len(data))
                
        elif self.method == 'ensemble':
            # Combine feature importance from multiple models
            feature_importance = self._calculate_ensemble_importance(X_scaled)
            
        else:
            # Fallback to permutation importance
            feature_importance = self._calculate_permutation_importance(X_scaled)
        
        return feature_importance
    
    def _calculate_permutation_importance(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calculate feature importance using permutation method.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Array of feature importance scores
        """
        base_score = self.predict_from_features(X_scaled)
        importances = np.zeros(X_scaled.shape[1])
        
        for i in range(X_scaled.shape[1]):
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self.predict_from_features(X_permuted)
            importances[i] = np.mean(np.abs(permuted_score - base_score))
        
        return importances
    
    def _calculate_gradient_importance(self, X_scaled: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate gradient-based feature importance for autoencoder.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Dictionary with feature importance scores
        """
        feature_importance = {}
        
        with tf.GradientTape() as tape:
            X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
            tape.watch(X_tensor)
            reconstructed = self.model(X_tensor)
            loss = tf.reduce_mean(tf.square(X_tensor - reconstructed), axis=1)
        
        gradients = tape.gradient(loss, X_tensor)
        gradients = gradients.numpy()
        
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = np.abs(gradients[:, i])
        
        return feature_importance
    
    def _calculate_ensemble_importance(self, X_scaled: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate ensemble feature importance.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Dictionary with feature importance scores
        """
        # Combine importance from multiple models
        importance_dicts = []
        
        # Isolation Forest importance
        if hasattr(self.models['isolation_forest'], 'feature_importances_'):
            if_importance = self.models['isolation_forest'].feature_importances_
        else:
            if_importance = self._calculate_permutation_importance(X_scaled)
        
        # PCA importance
        loadings = self.models['pca'].components_
        explained_variance = self.models['pca'].explained_variance_ratio_
        pca_importance = np.sum(np.abs(loadings) * explained_variance[:, np.newaxis], axis=0)
        
        # Autoencoder importance
        ae_importance = self._calculate_gradient_importance(X_scaled)
        
        # Combine all importances
        combined_importance = {}
        for feature in self.feature_names:
            feature_idx = self.feature_names.index(feature)
            if_score = if_importance[feature_idx]
            pca_score = pca_importance[feature_idx]
            ae_score = np.mean(ae_importance[feature])
            
            # Normalize and combine
            combined_score = (if_score + pca_score + ae_score) / 3
            combined_importance[feature] = combined_score * np.ones(len(X_scaled))
        
        return combined_importance
    
    def predict_from_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores from scaled feature matrix.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Array of anomaly scores
        """
        if self.method == 'isolation_forest':
            return -self.model.decision_function(X_scaled)
        elif self.method == 'autoencoder':
            reconstructed = self.model.predict(X_scaled)
            return np.sum((X_scaled - reconstructed) ** 2, axis=1)
        elif self.method == 'pca':
            reconstructed = self.model.inverse_transform(self.model.transform(X_scaled))
            return np.sum((X_scaled - reconstructed) ** 2, axis=1)
        else:
            raise ValueError(f"Method {self.method} not supported for feature prediction")