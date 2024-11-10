from data_pipeline import DataPipeline
from enhanced_knowledge import EnhancedKnowledgeBase
from unused.knowledge_transfer import KnowledgeBase
from learning_core import LearningMetricsTracker, ModelEnsemble
from safety_validator import SafetyValidator
from system_config import SystemConfig



class CoreSystem:
    def __init__(self, config: SystemConfig):
        # Data Processing Layer
        self.data_pipeline = DataPipeline(
            window_size=config.window_size,
            sampling_rate=config.sampling_rate
        )

        
        # Learning Core
        self.model_ensemble = ModelEnsemble(
            base_models=['rf', 'isolation_forest'],
            n_estimators=3
        )
        
        # Safety Framework
        self.safety_validator = SafetyValidator(
            physics_rules=PUMP_PHYSICS_RULES
        )
        
        
        self.model_ensemble = ModelEnsemble(
            base_models=['rf', 'isolation_forest'],
            n_estimators=3
        )
        self.model_ensemble.learning_metrics = LearningMetricsTracker(
            window_size=config.window_size
        )
            
        self.safety_validator = SafetyValidator(
            physics_rules=PUMP_PHYSICS_RULES
        )
        
        # Knowledge Management
        self.knowledge_base = EnhancedKnowledgeBase(
            nim_endpoint=config.nim_endpoint
        )

        # Monitoring
        #self.performance_monitor = PerformanceMonitor()

PUMP_PHYSICS_RULES = {
    'max_temperature': 90.0,  # Celsius
    'max_vibration': 12.0,    # mm/s
    'max_speed': 3800,        # RPM
    'rate_limits': {
        'temperature': 20.0,   # C/min - much more lenient
        'vibration': 5.0,     # mm/s/sec - much more lenient
        'speed': 500          # RPM/sec - much more lenient
    }
}