from dataclasses import dataclass, field
from typing import Dict

@dataclass
class SystemConfig:
    window_size: int = 100
    sampling_rate: float = 1.0
    nim_endpoint: str = "https://api.nvidia.com/nim/v1/models/nvolveqa_40k"
    learning_metrics_window: int = 100  
    safety_limits: Dict = field(default_factory=lambda: {
        'temperature': {'max': 75.0, 'min': 35.0, 'rate': 2.0},
        'vibration': {'max': 5.0, 'min': 0.0, 'rate': 1.0},
        'speed': {'max': 3500, 'min': 2500, 'rate': 100}
    })