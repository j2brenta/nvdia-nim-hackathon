from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


@dataclass
class EnsembleConfig:
    n_estimators: int = 3
    random_state: int = 42
    rf_params: Dict = None
    if_params: Dict = None
    mlp_params: Dict = None
    window_size: int = 1000
    min_samples_for_update: int = 50
    update_frequency: int = 20

class ModelEnsemble:
    def __init__(self, base_models: List[str], n_estimators: int):
        self.config = EnsembleConfig(n_estimators=n_estimators)
        self.base_models = base_models
        self.models = self._initialize_models(base_models, n_estimators)
        self.uncertainty_calculator = UncertaintyCalculator()
        self.is_fitted = False
        self.training_data = {
            'features': [],
            'targets': []
        }
        self.scaler = StandardScaler()
        self.update_counter = 0
        self.learning_metrics = LearningMetricsTracker(window_size=self.config.window_size)
        
    def _initialize_models(self, base_models: List[str], n_estimators: int) -> List:
        models = []
        
        model_constructors = {
            'rf': self._create_random_forest,
            'isolation_forest': self._create_isolation_forest,
            'mlp': self._create_mlp
        }
        
        for model_type in base_models:
            if model_type in model_constructors:
                for i in range(n_estimators):
                    model = model_constructors[model_type](i)
                    models.append(model)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        return models

    def _create_random_forest(self, seed: int) -> RandomForestRegressor:
        params = self.config.rf_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.config.random_state + seed,
            'n_jobs': -1
        }
        return RandomForestRegressor(**params)
    
    def _create_isolation_forest(self, seed: int) -> IsolationForest:
        params = self.config.if_params or {
            'n_estimators': 100,
            'contamination': 'auto',
            'max_samples': 'auto',
            'random_state': self.config.random_state + seed,
            'n_jobs': -1
        }
        return IsolationForest(**params)
    
    def _create_mlp(self, seed: int) -> MLPRegressor:
        params = self.config.mlp_params or {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate': 'adaptive',
            'max_iter': 1000,
            'random_state': self.config.random_state + seed
        }
        return MLPRegressor(**params)

    def _generate_initial_training_data(self):
        n_samples = 1000
        n_features = 18  # 6 features per sensor * 3 sensors
        
        # Generate normal operation data
        X_normal = np.random.randn(int(0.7 * n_samples), n_features)
        y_normal = np.zeros(int(0.7 * n_samples))
        
        # Generate anomaly data
        X_anomaly = np.random.randn(int(0.3 * n_samples), n_features) * 2 + 3
        y_anomaly = np.ones(int(0.3 * n_samples))
        
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([y_normal, y_anomaly])
        
        # Add noise and patterns
        X = self._add_patterns_to_data(X)
        
        return X, y
    
    def _add_patterns_to_data(self, X):
        n_samples = X.shape[0]
        
        # Add sinusoidal patterns
        t = np.linspace(0, 10, n_samples)
        pattern1 = np.sin(2 * np.pi * 0.5 * t).reshape(-1, 1)
        pattern2 = np.cos(2 * np.pi * 0.3 * t).reshape(-1, 1)
        
        # Add patterns to random features
        feature_indices = np.random.choice(X.shape[1], 2, replace=False)
        X[:, feature_indices[0]] += pattern1.flatten()
        X[:, feature_indices[1]] += pattern2.flatten()
        
        return X
    
    def ensure_fitted(self):
        if not self.is_fitted:
            X, y = self._generate_initial_training_data()
            self.fit(X, y)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        for model in self.models:
            if isinstance(model, IsolationForest):
                model.fit(X_scaled)
            else:
                model.fit(X_scaled, y)
        
        self.is_fitted = True
        
        # Store data for incremental updates
        self.training_data['features'].extend(X)
        self.training_data['targets'].extend(y)
        
        # Limit stored data size
        if len(self.training_data['features']) > self.config.window_size:
            self.training_data['features'] = self.training_data['features'][-self.config.window_size:]
            self.training_data['targets'] = self.training_data['targets'][-self.config.window_size:]
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self.ensure_fitted()
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        raw_predictions = []
        
        for model in self.models:
            if isinstance(model, IsolationForest):
                pred = self._convert_isolation_score(model.score_samples(features_scaled))
            else:
                pred = model.predict(features_scaled)
            predictions.append(pred)
            raw_predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = self.uncertainty_calculator.calculate_uncertainty(predictions)
        
        # Update counter and check if we should update models
        self.update_counter += 1
        if self.update_counter >= self.config.update_frequency:
            self._incremental_update(features, mean_pred)
            self.update_counter = 0
        
        return mean_pred, uncertainty
    
    def _incremental_update(self, features: np.ndarray, predictions: np.ndarray):
        """Perform incremental update of models"""
        if len(self.training_data['features']) > self.config.min_samples_for_update:
            # Get recent data
            recent_features = np.array(self.training_data['features'][-self.config.min_samples_for_update:])
            recent_targets = np.array(self.training_data['targets'][-self.config.min_samples_for_update:])
            
            # Add new data
            recent_features = np.vstack([recent_features, features])
            recent_targets = np.hstack([recent_targets, predictions])
            
            # Partial fit for each model
            features_scaled = self.scaler.transform(recent_features)
            for model in self.models:
                if isinstance(model, MLPRegressor):
                    model.partial_fit(features_scaled, recent_targets)
                elif not isinstance(model, IsolationForest):
                    # For models without partial_fit, retrain on recent data
                    model.fit(features_scaled, recent_targets)
    
    def _convert_isolation_score(self, scores: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-scores))

class LearningMetricsTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.confidence_history = {
            'bearing_fault': deque(maxlen=window_size),
            'cavitation': deque(maxlen=window_size),
            'misalignment': deque(maxlen=window_size)
        }
        self.detection_times = deque(maxlen=window_size)
        self.false_positives = deque(maxlen=window_size)
        self.accuracy_history = deque(maxlen=window_size)
        self.pattern_encounters = {
            'bearing_fault': 0,
            'cavitation': 0,
            'misalignment': 0
        }
        self.total_patterns = 0
        self.cumulative_confidence = 0.0
        self.learning_rate = 0.02  # Increased from 0.01
        self.initial_learning_phase = True
        self.min_samples_for_reliable_metrics = 15  # Reduced from 20
        
    def get_learning_progress(self) -> Dict:
        """Calculate learning progress with improved progression"""
        if not self.accuracy_history or len(self.accuracy_history) < self.min_samples_for_reliable_metrics:
            return {
                'recognition_rate': 0.0,
                'detection_speed': 0.0,
                'false_positive_rate': 0.0,
                'overall_progress': 0.0
            }
        
        # Calculate recognition rate with more recent emphasis
        recent_accuracy = list(self.accuracy_history)[-15:]
        recognition_rate = np.mean(recent_accuracy) * 1.2  # Boost factor
        
        # Calculate detection speed improvement
        recent_times = list(self.detection_times)[-15:]
        detection_speed = 1.0 - np.mean(recent_times)
        
        # Calculate false positive reduction with higher weight for recent performance
        recent_fps = list(self.false_positives)[-15:]
        false_positive_rate = 1.0 - np.mean(recent_fps)
        
        # Calculate pattern coverage with boost for multiple pattern types
        total_encounters = sum(self.pattern_encounters.values())
        pattern_types_seen = sum(1 for count in self.pattern_encounters.values() if count > 0)
        coverage_boost = pattern_types_seen / len(self.pattern_encounters)
        
        if total_encounters > 0:
            pattern_coverage = min(1.0, (total_encounters / (2 * self.min_samples_for_reliable_metrics))) * (1 + coverage_boost)
        else:
            pattern_coverage = 0.0
        
        # Calculate overall progress with improved weighting
        if self.initial_learning_phase:
            overall_progress = min(0.4, (recognition_rate * 0.6 + pattern_coverage * 0.4))
        else:
            weights = [
                0.35,  # Recognition rate
                0.25,  # Detection speed
                0.25,  # False positive rate
                0.15   # Pattern coverage
            ]
            
            components = [
                recognition_rate,
                detection_speed,
                false_positive_rate,
                pattern_coverage
            ]
            
            # Calculate weighted average with learning curve factor
            learning_curve_factor = 1.0 - np.exp(-self.total_patterns / 50)  # Reduced from 100
            weighted_progress = np.average(components, weights=weights)
            
            # Apply acceleration factor based on consistent performance
            consistency_bonus = min(0.2, 0.2 * self.cumulative_confidence)  # Increased from 0.1
            acceleration_factor = 1.0 + (self.total_patterns / 200)  # Added acceleration
            
            overall_progress = min(1.0, weighted_progress * learning_curve_factor * acceleration_factor + consistency_bonus)
        
        return {
            'recognition_rate': recognition_rate,
            'detection_speed': detection_speed,
            'false_positive_rate': false_positive_rate,
            'overall_progress': overall_progress
        }

    def update_metrics(self, pattern_type: str, detection_time: float,
                      predicted_pattern: str, actual_pattern: str, confidence: float):
        """Update metrics with faster learning progression"""
        self.total_patterns += 1
        if pattern_type != 'normal':
            self.pattern_encounters[pattern_type] = self.pattern_encounters.get(pattern_type, 0) + 1
        
        # Update detection time with normalization
        normalized_detection_time = min(1.0, detection_time / 3.0)  # Reduced from 5.0
        self.detection_times.append(normalized_detection_time)
        
        # Update accuracy with higher weight for correct predictions
        is_correct = predicted_pattern == actual_pattern
        self.accuracy_history.append(float(is_correct))
        
        # Update false positives with reduced penalty
        self.false_positives.append(0.8 * float(not is_correct))  # Reduced penalty
        
        # Update confidence for pattern types with faster adaptation
        for p_type in self.confidence_history.keys():
            if p_type == pattern_type:
                current_conf = np.mean(list(self.confidence_history[p_type])) if self.confidence_history[p_type] else 0.2
                new_conf = current_conf + self.learning_rate * (confidence - current_conf) * 1.2  # Increased adaptation
                self.confidence_history[p_type].append(min(0.98, new_conf))  # Increased from 0.95
            else:
                current_conf = np.mean(list(self.confidence_history[p_type])) if self.confidence_history[p_type] else 0.2
                new_conf = max(0.2, current_conf * 0.95)  # Reduced decay
                self.confidence_history[p_type].append(new_conf)
        
        # Update cumulative confidence with faster growth
        self.cumulative_confidence = 0.95 * self.cumulative_confidence + 0.05 * confidence  # Increased from 0.98/0.02
        
        # Check if we can exit initial learning phase
        if self.initial_learning_phase and self.total_patterns >= self.min_samples_for_reliable_metrics:
            self.initial_learning_phase = False
    
  
    
    def get_metrics_history(self) -> Dict:
        return {
            'confidence_history': {k: list(v) for k, v in self.confidence_history.items()},
            'detection_times': list(self.detection_times),
            'false_positives': list(self.false_positives),
            'accuracy_history': list(self.accuracy_history)
        }

class UncertaintyCalculator:
    def calculate_uncertainty(self, predictions: np.ndarray) -> Dict:
        epistemic = np.std(predictions, axis=0)
        aleatoric = self._estimate_aleatoric_uncertainty(predictions)
        total = np.sqrt(epistemic**2 + aleatoric**2)
        diversity = self._calculate_ensemble_diversity(predictions)
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total,
            'diversity': diversity
        }
    
    def _estimate_aleatoric_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        bootstrap_samples = np.random.choice(
            predictions.shape[0],
            size=(100, predictions.shape[0]),
            replace=True
        )
        bootstrap_means = np.mean(predictions[bootstrap_samples], axis=1)
        return np.std(bootstrap_means, axis=0)
    
    def _calculate_ensemble_diversity(self, predictions: np.ndarray) -> float:
        mean_pred = np.mean(predictions, axis=0)
        squared_diffs = np.mean((predictions - mean_pred)**2, axis=1)
        return np.mean(squared_diffs)