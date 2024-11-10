from collections import deque
from typing import Dict, List
import numpy as np
from scipy import stats


class DataPipeline:
    def __init__(self, window_size: int, sampling_rate: int):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.feature_extractor = FeatureExtractor(
            window_size=window_size,
            sampling_rate=sampling_rate
        )
        # Initialize deque buffers for each sensor
        self.buffers = {
            'temperature': deque(maxlen=window_size),
            'vibration': deque(maxlen=window_size),
            'speed': deque(maxlen=window_size)
        }
    
    def process_data(self, temperature: float, vibration: float, speed: float) -> Dict:
        """
        Process single data point with individual sensor values
        """
        # Update buffers with new values
        self.buffers['temperature'].append(float(temperature))
        self.buffers['vibration'].append(float(vibration))
        self.buffers['speed'].append(float(speed))
        
        # Convert current values to numpy array
        current_values = np.array([
            float(temperature),
            float(vibration),
            float(speed)
        ], dtype=np.float64)
        
        # Extract features from current values
        features = self.feature_extractor.extract_features(current_values)
        
        # Calculate basic statistics from buffer
        statistics = self._calculate_statistics()
        
        return {
            'features': features,
            'statistics': statistics,
            'current_values': {
                'temperature': temperature,
                'vibration': vibration,
                'speed': speed
            }
        }
    
    def _calculate_statistics(self) -> Dict:
        """Calculate statistics from buffered data"""
        stats = {}
        for sensor, buffer in self.buffers.items():
            if len(buffer) > 0:
                data = np.array(list(buffer))
                stats[sensor] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                }
            else:
                stats[sensor] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
        return stats

    
    def preprocess_raw_data(self, raw_data: Dict) -> np.ndarray:
        """Preprocess raw sensor data into format for feature extraction"""
        # Combine sensor readings into single array
        sensor_data = np.array([
            raw_data['temperature'],
            raw_data['vibration'],
            raw_data['speed']
        ])
        
        # Normalize data
        normalized_data = self._normalize_data(sensor_data)
        
        return normalized_data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using running statistics"""
        if not self.data_buffer:
            return data
        
        buffer_array = np.array(self.data_buffer)
        mean = np.mean(buffer_array, axis=0)
        std = np.std(buffer_array, axis=0) + 1e-8  # Avoid division by zero
        
        return (data - mean) / std
    
class FeatureExtractor:
    def __init__(self, window_size: int, sampling_rate: int):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
        
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from single data point"""
        features = []
        
        # Process each sensor value
        for sensor_value in data:
            # Time domain features for single value
            features.extend([
                float(sensor_value),  # Current value
                0,  # No std for single value
                float(sensor_value),  # Max is same as value
                float(sensor_value),  # Min is same as value
                0,  # No skew for single value
                0   # No kurtosis for single value
            ])
        
        return np.array(features, dtype=np.float64)
    
    def _time_domain_single(self, data: np.ndarray) -> List[float]:
        """Calculate time domain features for a single sensor"""
        return [
            np.mean(data),
            np.std(data),
            np.max(data),
            np.min(data),
            stats.skew(data),
            stats.kurtosis(data)
        ]
    
    def _frequency_domain_single(self, data: np.ndarray) -> List[float]:
        """Calculate frequency domain features for a single sensor"""
        # Apply window function
        window = np.hanning(len(data))
        windowed_data = data * window
        
        # Compute FFT
        fft_vals = np.abs(np.fft.fft(windowed_data))
        freqs = np.fft.fftfreq(len(data), 1/self.sampling_rate)
        
        # Only consider positive frequencies up to Nyquist
        positive_freq_mask = (freqs >= 0) & (freqs <= self.nyquist_freq)
        fft_vals = fft_vals[positive_freq_mask]
        freqs = freqs[positive_freq_mask]
        
        return [
            np.max(fft_vals),
            np.mean(fft_vals),
            np.std(fft_vals),
            freqs[np.argmax(fft_vals)],  # Peak frequency
            np.sum(freqs * fft_vals) / np.sum(fft_vals)  # Spectral centroid
        ]