from typing import Dict, List, Tuple
from collections import deque
from datetime import datetime

import numpy as np
class SafetyValidator:
    def __init__(self, physics_rules: Dict):
        self.rules = physics_rules
        self.history = {
            'temperature': deque(maxlen=10),
            'vibration': deque(maxlen=10),
            'speed': deque(maxlen=10),
            'timestamps': deque(maxlen=10)
        }
        self.violation_history = []
        
        # Add normal operation ranges
        self.normal_ranges = {
            'temperature': (35.0, 70.0),   # Wider temperature range
            'vibration': (0.2, 4.0),       # Wider vibration range
            'speed': (2700, 3300)          # Wider speed range
        }
        
        # Add warning thresholds
        self.warning_thresholds = {
            'temperature': self.rules['max_temperature'] * 0.9,
            'vibration': self.rules['max_vibration'] * 0.9,
            'speed': self.rules['max_speed'] * 0.9
        }
        
        # Moving average for smoother rate calculations
        self.rate_history = {
            'temperature': deque(maxlen=5),
            'vibration': deque(maxlen=5),
            'speed': deque(maxlen=5)
        }
    
    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate sensor data against safety rules
        Args:
            data: Dictionary with 'temperature', 'vibration', and 'speed' values
        Returns:
            Tuple of (is_safe, list of violations)
        """
        violations = []
        warnings = []
        
        # Update history
        current_time = datetime.now()
        for sensor in ['temperature', 'vibration', 'speed']:
            self.history[sensor].append(data[sensor])
        self.history['timestamps'].append(current_time)
        
        # Check absolute limits only if significantly outside normal range
        for sensor in ['temperature', 'vibration', 'speed']:
            current_value = data[sensor]
            normal_range = self.normal_ranges[sensor]
            
            # Calculate how far outside normal range (if at all)
            if current_value < normal_range[0]:
                deviation = (normal_range[0] - current_value) / normal_range[0]
            elif current_value > normal_range[1]:
                deviation = (current_value - normal_range[1]) / normal_range[1]
            else:
                deviation = 0
            
            # Only check absolute limits if significantly outside normal range
            if deviation > 0.2:  # 20% deviation threshold
                if current_value > self.rules[f'max_{sensor}']:
                    violations.append(
                        f"{sensor.capitalize()} exceeded: {current_value:.1f} > {self.rules[f'max_{sensor}']}"
                    )
                elif current_value > self.warning_thresholds[sensor]:
                    warnings.append(
                        f"{sensor.capitalize()} warning: {current_value:.1f}"
                    )
        
        # Check rate of change with more lenient thresholds
        if len(self.history['timestamps']) >= 2:
            rate_violations = self._check_rate_violations()
            if rate_violations:
                violations.extend(rate_violations)
        
        # Store violation in history if any
        if violations:
            self.violation_history.append({
                'timestamp': current_time,
                'violations': violations,
                'data': data.copy()
            })
            if len(self.violation_history) > 100:
                self.violation_history.pop(0)
        
        is_safe = len(violations) == 0
        return is_safe, violations if not is_safe else warnings
    
    def _check_rate_violations(self) -> List[str]:
        """Check for rate of change violations with smoothing"""
        violations = []
        
        if len(self.history['timestamps']) < 2:
            return violations
        
        # Calculate time difference in minutes
        time_diff = (self.history['timestamps'][-1] - self.history['timestamps'][-2]).total_seconds() / 60.0
        
        # Skip rate check if time difference is too small
        if time_diff < 0.001:
            return violations
        
        for sensor in ['temperature', 'vibration', 'speed']:
            if len(self.history[sensor]) < 2:
                continue
            
            # Calculate rate of change
            value_diff = abs(self.history[sensor][-1] - self.history[sensor][-2])
            rate = value_diff / time_diff
            
            # Add to rate history
            self.rate_history[sensor].append(rate)
            
            # Use moving average for smoother rate calculation
            avg_rate = np.mean(list(self.rate_history[sensor]))
            
            # Get appropriate rate limit based on operating zone
            base_rate_limit = self.rules['rate_limits'][sensor]
            
            # Determine multiplier based on current value's position in range
            current_value = self.history[sensor][-1]
            normal_range = self.normal_ranges[sensor]
            
            if normal_range[0] <= current_value <= normal_range[1]:
                # In normal range - use very lenient limits
                rate_limit = base_rate_limit * 10
            else:
                # Outside normal range - use standard limits
                rate_limit = base_rate_limit * 2
            
            # Only violate if average rate exceeds limit significantly
            if avg_rate > rate_limit:
                violations.append(
                    f"{sensor.capitalize()} changing too fast: {avg_rate:.2f} per minute > {rate_limit:.2f}"
                )
        
        return violations
    
    def get_recent_violations(self, n: int = 5) -> List[Dict]:
        """Get the n most recent violations"""
        return self.violation_history[-n:]