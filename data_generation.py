# data_generator.py
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class PumpParameters:
    normal_temperature_range: Tuple[float, float] = (40.0, 60.0)
    normal_vibration_range: Tuple[float, float] = (1.0, 3.0)
    normal_speed_range: Tuple[float, float] = (2800, 3200)
    
    # Fault patterns
    bearing_fault_vib_increase: float = 2.5
    cavitation_pattern_freq: float = 0.1
    misalignment_temp_increase: float = 15.0
    
    # Time constants
    sampling_rate: float = 100  # Hz
    pattern_duration: float = 300  # seconds

class PumpDataGenerator:
    def __init__(self, params: PumpParameters):
        self.params = params
        self.time = 0
        self.current_state = 'normal'
        self.fault_types = ['bearing_fault', 'cavitation', 'misalignment']
        
    def generate_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        if np.random.random() < 0.05:  # 5% chance of fault
            self.current_state = np.random.choice(self.fault_types)
        
        data = {
            'timestamp': np.arange(self.time, self.time + batch_size) / self.params.sampling_rate,
            'temperature': self._generate_temperature(batch_size),
            'vibration': self._generate_vibration(batch_size),
            'speed': self._generate_speed(batch_size)
        }
        
        self.time += batch_size
        return data
    
    def _generate_temperature(self, size: int) -> np.ndarray:
        base_temp = np.random.uniform(*self.params.normal_temperature_range)
        temp = base_temp + np.random.normal(0, 0.5, size)
        
        if self.current_state == 'misalignment':
            temp += self.params.misalignment_temp_increase
        
        return temp
    
    def _generate_vibration(self, size: int) -> np.ndarray:
        base_vib = np.random.uniform(*self.params.normal_vibration_range)
        vib = base_vib + np.random.normal(0, 0.2, size)
        
        if self.current_state == 'bearing_fault':
            vib += self.params.bearing_fault_vib_increase
        elif self.current_state == 'cavitation':
            vib += np.sin(2 * np.pi * self.params.cavitation_pattern_freq * 
                         np.arange(size)) * 2
        
        return vib
    
    def _generate_speed(self, size: int) -> np.ndarray:
        base_speed = np.random.uniform(*self.params.normal_speed_range)
        return base_speed + np.random.normal(0, 50, size)

