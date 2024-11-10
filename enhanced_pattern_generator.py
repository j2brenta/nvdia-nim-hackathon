from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np

def default_bearing_fault() -> Dict:
    return {
        'frequencies': [50, 100, 150],  # Hz
        'amplitudes': [1.5, 1.0, 0.5],  # Relative amplitudes
        'duration': 1000,  # samples
        'noise_level': 0.2
    }

def default_cavitation() -> Dict:
    return {
        'frequency_range': (500, 1000),  # Hz
        'modulation_freq': 10,  # Hz
        'duration': 800,  # samples
        'intensity': 1.5,
        'noise_level': 0.3
    }

def default_misalignment() -> Dict:
    return {
        'temperature_increase': 15.0,  # Celsius
        'vibration_amplitude': 2.0,  # mm/s
        'frequency': 1.0,  # Hz (for vibration pattern)
        'duration': 1200,  # samples
        'noise_level': 0.15
    }

@dataclass
class PatternConfig:
    # Fault pattern configurations using factory functions
    bearing_fault: Dict = field(default_factory=default_bearing_fault)
    cavitation: Dict = field(default_factory=default_cavitation)
    misalignment: Dict = field(default_factory=default_misalignment)
    
    # General settings
    sampling_rate: float = 1000.0  # Hz
    base_temperature: float = 45.0  # Celsius
    base_vibration: float = 2.0  # mm/s
    base_speed: float = 3000.0  # RPM

class EnhancedPatternGenerator:
    def __init__(self, config: PatternConfig):
        self.config = config
        self.current_state = 'normal'
        self.time = 0
        self.state_duration = 0
        self.time_in_state = 0
        self.min_state_duration = 50
        self.max_normal_duration = 500
        self.max_fault_duration = 200
        
        self.base_transition_probs = {
            'normal': {
                'stay_normal': 0.98,
                'bearing_fault': 0.007,
                'cavitation': 0.007,
                'misalignment': 0.006
            },
            'bearing_fault': {'normal': 0.1},
            'cavitation': {'normal': 0.15},
            'misalignment': {'normal': 0.05}
        }
        
        self.duration_threshold = 100
    
    def generate_pattern(self, pattern_type: Optional[str] = None, duration: Optional[int] = None) -> Dict:
        """Generate pattern with enhanced variation and realism"""
        if pattern_type is None:
            pattern_type = self.current_state
            
        if duration is None:
            duration = self.config.bearing_fault['duration']
            
        t = np.linspace(0, duration / self.config.sampling_rate, duration)
        time_factor = np.sin(2 * np.pi * 0.01 * self.time)
        
        if pattern_type == 'bearing_fault':
            data = self._generate_enhanced_bearing_fault(t, time_factor)
        elif pattern_type == 'cavitation':
            data = self._generate_enhanced_cavitation(t, time_factor)
        elif pattern_type == 'misalignment':
            data = self._generate_enhanced_misalignment(t, time_factor)
        else:
            data = self._generate_enhanced_normal(t, time_factor)
            
        # Ensure all data is returned as numpy arrays
        return {
            'temperature': np.atleast_1d(data['temperature']),
            'vibration': np.atleast_1d(data['vibration']),
            'speed': np.atleast_1d(data['speed']),
            'timestamp': data['timestamp'],
            'pattern_type': data['pattern_type']
        }
    
    def _generate_enhanced_normal(self, t: np.ndarray, time_factor: float) -> Dict:
        """Generate normal operation with subtle variations"""
        baseline_shift = 0.1 * time_factor
        
        # Generate all signals as arrays
        noise_level = 0.1 + 0.05 * np.sin(2 * np.pi * 0.02 * self.time)
        vibration = np.full(len(t), self.config.base_vibration * (1 + baseline_shift))
        vibration += noise_level * np.random.normal(0, 1, size=len(t))
        
        temp_variation = 1.0 + 0.5 * np.sin(2 * np.pi * 0.01 * self.time)
        temperature = np.full(len(t), self.config.base_temperature + temp_variation + baseline_shift)
        
        speed_variation = 20 * (1 + 0.2 * np.sin(2 * np.pi * 0.03 * self.time))
        speed = np.full(len(t), self.config.base_speed)
        speed += speed_variation * np.random.normal(0, 1, size=len(t))
        
        return {
            'vibration': vibration,
            'temperature': temperature,
            'speed': speed,
            'timestamp': datetime.now(),
            'pattern_type': 'normal'
        }
    
    def _generate_enhanced_bearing_fault(self, t: np.ndarray, time_factor: float) -> Dict:
        """Generate bearing fault with dynamic characteristics"""
        freq_mod = 1 + 0.1 * time_factor
        vibration = np.zeros(len(t))
        
        for freq, amp in zip(
            self.config.bearing_fault['frequencies'],
            self.config.bearing_fault['amplitudes']
        ):
            vibration += amp * np.sin(2 * np.pi * freq * freq_mod * t)
        
        harmonic_factor = 0.3 + 0.1 * np.sin(2 * np.pi * 0.05 * self.time)
        for freq in self.config.bearing_fault['frequencies']:
            vibration += harmonic_factor * np.sin(4 * np.pi * freq * t)
        
        vibration += np.random.normal(
            0,
            self.config.bearing_fault['noise_level'] * (1 + 0.2 * time_factor),
            size=len(t)
        )
        
        base_vibration = self.config.base_vibration * (1.2 + 0.2 * time_factor)
        vibration = base_vibration * (1 + vibration)
        
        temp_increase = 5 * (1 - np.exp(-self.time_in_state/100))
        temperature = np.full(len(t), self.config.base_temperature + temp_increase + 2 * time_factor)
        
        speed_variation = 50 * (1 + 0.5 * self.time_in_state/100)
        speed = self.config.base_speed + speed_variation * np.sin(2 * np.pi * 0.1 * t)
        
        return {
            'vibration': vibration,
            'temperature': temperature,
            'speed': speed,
            'timestamp': datetime.now(),
            'pattern_type': 'bearing_fault'
        }

    def check_state_transition(self) -> str:
        """Enhanced state transition logic with duration-based probabilities"""
        self.time_in_state += 1
        
        if self.current_state == 'normal':
            duration_factor = min(3.0, 1.0 + (self.time_in_state / self.duration_threshold))
            
            probs = {
                'stay_normal': max(0.90, self.base_transition_probs['normal']['stay_normal'] / duration_factor),
                'bearing_fault': self.base_transition_probs['normal']['bearing_fault'] * duration_factor,
                'cavitation': self.base_transition_probs['normal']['cavitation'] * duration_factor,
                'misalignment': self.base_transition_probs['normal']['misalignment'] * duration_factor
            }
            
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            
            random_val = np.random.random()
            cumulative_prob = 0
            
            for state, prob in probs.items():
                cumulative_prob += prob
                if random_val <= cumulative_prob:
                    if state != 'stay_normal':
                        self.current_state = state
                        self.time_in_state = 0
                        self.state_duration = np.random.randint(
                            self.min_state_duration,
                            self.max_fault_duration
                        )
                    break
                    
        else:
            if self.time_in_state >= self.state_duration:
                transition_prob = self.base_transition_probs[self.current_state]['normal']
                
                if np.random.random() < transition_prob:
                    self.current_state = 'normal'
                    self.time_in_state = 0
                    self.state_duration = np.random.randint(
                        self.min_state_duration,
                        self.max_normal_duration
                    )
        
        return self.current_state
    
    def _generate_enhanced_misalignment(self, t: np.ndarray, time_factor: float) -> Dict:
        """Generate misalignment pattern with dynamic characteristics"""
        # Strong periodic vibration with harmonics
        vibration = np.zeros(len(t))
        
        # Primary vibration component
        base_freq = self.config.misalignment['frequency']
        vibration += self.config.misalignment['vibration_amplitude'] * np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics
        harmonic_factor = 0.4 + 0.1 * np.sin(2 * np.pi * 0.03 * self.time)
        vibration += harmonic_factor * np.sin(4 * np.pi * base_freq * t)  # 2nd harmonic
        vibration += 0.3 * harmonic_factor * np.sin(6 * np.pi * base_freq * t)  # 3rd harmonic
        
        # Add noise with time-varying amplitude
        vibration += np.random.normal(
            0,
            self.config.misalignment['noise_level'] * (1 + 0.2 * time_factor),
            size=len(t)
        )
        
        # Scale vibration
        base_vibration = self.config.base_vibration * (1.3 + 0.3 * time_factor)
        vibration = base_vibration * (1 + vibration)
        
        # Temperature increases significantly with fault progression
        temp_increase = self.config.misalignment['temperature_increase'] * (1 - np.exp(-self.time_in_state/200))
        temperature = np.full(len(t), self.config.base_temperature + temp_increase + 3 * time_factor)
        
        # Speed shows periodic variations increasing with severity
        speed_variation = 150 * (1 + 0.3 * self.time_in_state/100)
        speed = self.config.base_speed + speed_variation * np.sin(2 * np.pi * 0.05 * t)
        
        return {
            'vibration': vibration,
            'temperature': temperature,
            'speed': speed,
            'timestamp': datetime.now(),
            'pattern_type': 'misalignment'
        }

    def _generate_enhanced_cavitation(self, t: np.ndarray, time_factor: float) -> Dict:
        """Generate cavitation pattern with dynamic characteristics"""
        # High-frequency modulated vibration
        carrier_freq = np.random.uniform(
            self.config.cavitation['frequency_range'][0],
            self.config.cavitation['frequency_range'][1]
        )
        
        # Create modulated signal with time-varying characteristics
        modulation_freq = self.config.cavitation['modulation_freq'] * (1 + 0.1 * time_factor)
        modulator = np.sin(2 * np.pi * modulation_freq * t)
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        
        # Base vibration signal
        vibration = (1 + modulator) * carrier
        
        # Add broadband noise component
        noise = np.random.normal(
            0,
            self.config.cavitation['noise_level'] * (1 + 0.3 * time_factor),
            size=len(t)
        )
        vibration += noise
        
        # Scale to physical units with temporal variation
        base_vibration = self.config.base_vibration * (1 + self.config.cavitation['intensity'] * (1 + 0.2 * time_factor))
        vibration = base_vibration * (1 + vibration)
        
        # Temperature increases moderately with fault progression
        temp_increase = 8 * (1 - np.exp(-self.time_in_state/150))
        temperature = np.full(len(t), self.config.base_temperature + temp_increase + 2 * time_factor)
        
        # Speed fluctuates more significantly with increased turbulence
        speed_variation = 100 * (1 + 0.4 * self.time_in_state/100)
        speed = self.config.base_speed + speed_variation * np.sin(2 * np.pi * 0.2 * t)
        
        # Add high-frequency components to speed due to flow instability
        speed += 20 * np.sin(2 * np.pi * 2 * t) * np.random.normal(0, 1, size=len(t))
        
        return {
            'vibration': vibration,
            'temperature': temperature,
            'speed': speed,
            'timestamp': datetime.now(),
            'pattern_type': 'cavitation'
        }        