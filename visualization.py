import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import deque
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Import system components
from core_system import CoreSystem, PUMP_PHYSICS_RULES
from data_generation import PumpDataGenerator, PumpParameters
from enhanced_pattern_generator import EnhancedPatternGenerator
from unused.pattern_generator import PatternConfig, PatternSequenceGenerator
from system_config import SystemConfig

class IntegratedSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.core_system = CoreSystem(config)
        self.current_state = 'normal'
        # Initialize and train the model ensemble
        self.core_system.model_ensemble.ensure_fitted()
        self.pattern_generator = EnhancedPatternGenerator(PatternConfig())
        self.sequence_generator = PatternSequenceGenerator(self.pattern_generator)

        self.viz_buffers = {
            'temperature': deque(maxlen=100),
            'vibration': deque(maxlen=100),
            'speed': deque(maxlen=100),
            'timestamp': deque(maxlen=100),
            'states': deque(maxlen=100),
            'violations': deque(maxlen=100),
            'pattern_type': []
        }
        self.max_buffer_size = 100
        self.uncertainty_buffer = deque(maxlen=config.window_size)
        self.pattern_buffer = deque(maxlen=config.window_size)
        # Track current simulation state
        self.current_state = 'normal'
        self.state_duration = 0
        self.time_in_state = 0
        
        # Pattern transition probabilities
        self.transition_probs = {
            'normal': 0.98,      # Probability to stay normal
            'fault': 0.02        # Probability to introduce fault
        }
        self.fault_types = ['bearing_fault', 'cavitation', 'misalignment']
    
    def _update_buffers(self, raw_data: Dict, is_safe: bool, violations: List[str], pattern_type: str):
            """Update visualization buffers with new data"""
            timestamp = datetime.now()
            
            # Add new data
            self.viz_buffers['timestamp'].append(timestamp)
            self.viz_buffers['temperature'].append(raw_data['temperature'])
            self.viz_buffers['vibration'].append(raw_data['vibration'])
            self.viz_buffers['speed'].append(raw_data['speed'])
            self.viz_buffers['states'].append(is_safe)
            self.viz_buffers['violations'].append(violations)
            self.viz_buffers['pattern_type'].append(pattern_type)
            
            # Maintain buffer size
            if len(self.viz_buffers['timestamp']) > self.max_buffer_size:
                for key in self.viz_buffers:
                    self.viz_buffers[key] = self.viz_buffers[key][-self.max_buffer_size:]

    def process_data_point(self) -> Tuple[Dict, List[float], Dict, bool, List[str]]:
        # Check for state transition
        self.check_state_transition()

        # Generate pattern based on current state
        pattern_data = self.pattern_generator.generate_pattern(
            self.current_state,
            duration=1
        )

        # Extract single data point from pattern data
        raw_data = {
            'temperature': float(pattern_data['temperature'][0]),
            'vibration': float(pattern_data['vibration'][0]),
            'speed': float(pattern_data['speed'][0])
        }

        # Process through data pipeline
        processed_data = self.core_system.data_pipeline.process_data(
            raw_data['temperature'],
            raw_data['vibration'],
            raw_data['speed']
        )
        
        # Get predictions
        features = np.array(processed_data['features']).reshape(1, -1)
        predictions, uncertainty = self.core_system.model_ensemble.predict(features)
        
        # Validate safety
        is_safe, violations = self.core_system.safety_validator.validate(raw_data)
        
        # Update knowledge base if safe
        if is_safe:
            pattern = {
                'features': processed_data['features'],
                'predictions': predictions,
                'uncertainty': uncertainty,
                'timestamp': datetime.now()
            }
            
            pattern_id = self.core_system.knowledge_base.add_pattern(pattern)
            similar_patterns = self.core_system.knowledge_base.find_similar_patterns(pattern)
            analysis = self.core_system.knowledge_base.analyze_pattern(pattern_id, similar_patterns)
            
            if analysis['trend']['type'] == 'degrading':
                print(f"Warning: Pattern quality degrading (confidence: {analysis['trend']['confidence']:.2f})")
            
            if analysis['anomaly_score'] > 0.8:
                print(f"Warning: Unusual pattern detected (anomaly score: {analysis['anomaly_score']:.2f})")
        
        # Update visualization buffers
        self._update_buffers(raw_data, is_safe, violations, self.current_state)
        
        return processed_data, predictions, uncertainty, is_safe, violations

    def check_state_transition(self):
            """Check and handle state transitions"""
            self.time_in_state += 1
            
            # Check for state transition
            if self.current_state == 'normal':
                if np.random.random() > self.transition_probs['normal']:
                    # Transition to fault
                    self.current_state = np.random.choice(self.fault_types)
                    self.time_in_state = 0
                    self.state_duration = np.random.randint(50, 200)  # Duration of fault
            else:
                if self.time_in_state >= self.state_duration:
                    # Return to normal
                    self.current_state = 'normal'
                    self.time_in_state = 0
                    self.state_duration = np.random.randint(200, 500)  # Duration of normal operation
    
def safe_float_conversion(value) -> float:
    """Safely convert numpy values to float"""
    if isinstance(value, np.ndarray):
        return float(value.item()) if value.size == 1 else float(value[0])
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    return float(value)


def main():
    # Initialize system
    config = SystemConfig()
    params = PumpParameters()
    data_generator = PumpDataGenerator(params)

    if 'system' not in st.session_state:
        st.session_state.system = IntegratedSystem(config)
        st.session_state.running = False
        st.session_state.simulation_speed = 1.0
    
    # Set page config to wide mode
    st.set_page_config(layout="wide")
    
    # Streamlit interface with more compact layout
    st.title("Pump Monitoring System")
    
    # Create two columns for the layout
    control_col, display_col = st.columns([1, 4])
    
    # Move controls to the left column
    with control_col:
        st.header("Controls")
        running = st.toggle('Start/Stop Simulation', st.session_state.running)
        simulation_speed = st.slider(
            'Simulation Speed',
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.simulation_speed,
            step=0.1
        )
        
        # if st.button('Force Fault'):
        #     st.session_state.system.current_state = np.random.choice(
        #         st.session_state.system.fault_types
        #     )
        #     st.session_state.system.time_in_state = 0
        #     st.session_state.system.state_duration = np.random.randint(50, 200)
        
        if st.button('Reset Simulation'):
            st.session_state.system = IntegratedSystem(SystemConfig())
            st.rerun()
    
    # Main display area
    with display_col:
        if running:
            start_time = time.time()
            
            # Process data point
            processed_data, predictions, uncertainty, is_safe, violations = st.session_state.system.process_data_point()
            
            # Update learning metrics
            detection_time = time.time() - start_time
            st.session_state.system.core_system.model_ensemble.learning_metrics.update_metrics(
                pattern_type=st.session_state.system.current_state,
                detection_time=detection_time,
                predicted_pattern=predictions[0],
                actual_pattern=st.session_state.system.current_state,
                confidence=1 - uncertainty['total'][0]
            )
            
            # Display current values and status in a single row
            cols = st.columns([1, 1, 1, 1])
            
            if 'previous_values' not in st.session_state:
                st.session_state.previous_values = {
                    'temperature': None,
                    'vibration': None,
                    'speed': None
                }

            # Calculate meaningful deltas using previous values instead of means
            with cols[0]:
                st.metric(
                    "Temperature",
                    f"{processed_data['current_values']['temperature']:.1f}°C",
                    f"{processed_data['current_values']['temperature'] - st.session_state.previous_values['temperature']:.1f}°C" 
                    if st.session_state.previous_values['temperature'] is not None else None
                )

            with cols[1]:
                st.metric(
                    "Vibration",
                    f"{processed_data['current_values']['vibration']:.2f} mm/s",
                    f"{processed_data['current_values']['vibration'] - st.session_state.previous_values['vibration']:.2f} mm/s"
                    if st.session_state.previous_values['vibration'] is not None else None
                )

            with cols[2]:
                st.metric(
                    "Speed",
                    f"{processed_data['current_values']['speed']:.0f} RPM",
                    f"{processed_data['current_values']['speed'] - st.session_state.previous_values['speed']:.0f} RPM"
                    if st.session_state.previous_values['speed'] is not None else None
                )
            
            with cols[3]:
                status_color = "🟢" if is_safe else "🔴"
                pattern_type = st.session_state.system.current_state
                st.metric("Status", f"{status_color} {pattern_type}")
            
            st.session_state.previous_values = {
                'temperature': processed_data['current_values']['temperature'],
                'vibration': processed_data['current_values']['vibration'],
                'speed': processed_data['current_values']['speed']
            }

            # Create two columns for charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                monitoring_fig = create_monitoring_charts(st.session_state.system.viz_buffers)
                # Reduce height for more compact view
                monitoring_fig.update_layout(height=500)
                st.plotly_chart(monitoring_fig, use_container_width=True)
            
            with chart_col2:
                metrics_history = st.session_state.system.core_system.model_ensemble.learning_metrics.get_metrics_history()
                learning_progress = st.session_state.system.core_system.model_ensemble.learning_metrics.get_learning_progress()
                learning_fig = create_learning_progress_charts(metrics_history, learning_progress)
                # Reduce height for more compact view
                learning_fig.update_layout(height=500)
                st.plotly_chart(learning_fig, use_container_width=True)
            
            # Create two columns for additional info
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                if violations:
                    st.warning("Safety Violations:\n" + "\n".join(violations))
                
                st.subheader("Pattern Information")
                # Only show current pattern info, not historical
                pattern_info = {
                    'Current State': st.session_state.system.current_state,
                    'Time in State': st.session_state.system.time_in_state,
                    'State Duration': st.session_state.system.state_duration
                }
                # Use native Streamlit formatting instead of raw dict display
                for key, value in pattern_info.items():
                    st.write(f"**{key}:** {value}")
            
            with info_col2:
                if hasattr(st.session_state.system.pattern_generator, 'analyze_generated_pattern'):
                    pattern = st.session_state.system.pattern_generator.generate_pattern(
                        st.session_state.system.current_state
                    )
                    analysis = st.session_state.system.pattern_generator.analyze_generated_pattern(pattern)
                    pattern_fig = create_pattern_visualization(analysis)
                    # Reduce height for more compact view
                    pattern_fig.update_layout(height=400)
                    st.plotly_chart(pattern_fig, use_container_width=True)
            
            # Sleep based on simulation speed
            time.sleep(1.0 / simulation_speed)
            st.rerun()

def create_monitoring_charts(buffers: Dict) -> go.Figure:
    """Create monitoring charts with pattern indicators"""
    # Convert deque objects to lists for Plotly
    timestamps = list(buffers['timestamp'])
    temperatures = list(buffers['temperature'])
    vibrations = list(buffers['vibration'])
    speeds = list(buffers['speed'])
    patterns = list(buffers['pattern_type'])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Temperature',
            'Vibration',
            'Speed',
            'Pattern Status'
        ),
        vertical_spacing=0.3,
        horizontal_spacing=0.15
    )
    
    # Temperature subplot
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=temperatures,
            name="Temperature",
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Vibration subplot
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=vibrations,
            name="Vibration",
            line=dict(color='blue', width=2)
        ),
        row=1, col=2
    )
    
    # Speed subplot
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=speeds,
            name="Speed",
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    # Pattern indicator
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[1 if s == 'normal' else 0 for s in patterns],
            name="Pattern",
            line=dict(color='purple', width=2),
            fill='tozeroy'
        ),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Vibration (mm/s)", row=1, col=2)
    fig.update_yaxes(title_text="Speed (RPM)", row=2, col=1)
    fig.update_yaxes(title_text="Pattern", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        width=900,
        height=600
    )
    
    return fig

def create_learning_progress_charts(metrics_history: Dict, learning_progress: Dict) -> go.Figure:
    """Create charts showing learning progress and metrics"""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "indicator"}]
        ],
        vertical_spacing=0.22,
        horizontal_spacing=0.15
    )
    
    # Pattern recognition confidence
    colors = {'bearing_fault': 'blue', 'cavitation': 'orange', 'misalignment': 'purple'}
    for pattern_type, confidence in metrics_history['confidence_history'].items():
        fig.add_trace(
            go.Scatter(
                y=list(confidence),
                name=pattern_type.replace('_', ' ').title(),
                mode='lines',
                line=dict(color=colors.get(pattern_type, 'gray'))
            ),
            row=1, col=1
        )
    
    # Detection speed improvement
    fig.add_trace(
        go.Scatter(
            y=metrics_history['detection_times'],
            name='Detection Time',
            mode='lines',
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # False positive rate
    fig.add_trace(
        go.Scatter(
            y=metrics_history['false_positives'],
            name='False Positive Rate',
            mode='lines',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    # Overall learning progress gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=learning_progress['overall_progress'] * 100,
            title={'text': "Learning Progress"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=2
    )
    
    # Update layout with adjusted margins, spacing, and legend
    fig.update_layout(
        height=800,
        showlegend=True,  # Enable legend
        margin=dict(l=20, r=100, t=150, b=20),  # Increased right margin for legend
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        legend=dict(
            yanchor="top",
            y=0.95,        # Position near top
            xanchor="left",
            x=1.05,        # Position to the right of the charts
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=12)
        ),
        annotations=[
            dict(
                text="Learning Metrics History",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.25,
                showarrow=False,
                font=dict(
                    size=28,
                    color="black",
                    family="Source Sans Pro"
                ),
                xanchor='center',
                yanchor='bottom'
            )
        ]
    )
    
    # Update subplot titles font and position
    for annotation in fig.layout.annotations[:-1]:  # Exclude the main title
        annotation.update(
            font=dict(size=14, color="black"),
            y=annotation.y + 0.05
        )
    
    # Update axes labels
    fig.update_yaxes(title_text="Confidence", row=1, col=1)
    fig.update_yaxes(title_text="Detection Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="False Positive Rate", row=2, col=1)
    
    return fig

def create_pattern_visualization(analysis_data: Dict) -> go.Figure:
    """
    Create visualization for pattern analysis with correctly specified subplot types
    Args:
        analysis_data: Dictionary containing pattern analysis results
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "xy"}]
        ],
        subplot_titles=(
            'Vibration Level',
            'Temperature',
            'Speed',
            'Frequency Spectrum'
        ),
        vertical_spacing=0.3,    # Increase spacing between rows
        horizontal_spacing=0.1,   # Add horizontal spacing
    )
    
    # Helper function to safely get values
    def safe_get(data, *keys, default=0.0):
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    # Vibration Analysis
    vibration_stats = safe_get(analysis_data, 'vibration', default={})
    if vibration_stats:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=safe_get(vibration_stats, 'mean', default=0.0),
                title={'text': "Vibration Level (mm/s)"},
                gauge={
                    'axis': {'range': [0, 10]},
                    'steps': [
                        {'range': [0, 3], 'color': "green"},
                        {'range': [3, 7], 'color': "yellow"},
                        {'range': [7, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': safe_get(vibration_stats, 'max', default=10.0)
                    }
                }
            ),
            row=1, col=1
        )

    # Temperature Analysis
    temp_stats = safe_get(analysis_data, 'temperature', default={})
    if temp_stats:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=safe_get(temp_stats, 'mean', default=0.0),
                title={'text': "Temperature (°C)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': safe_get(temp_stats, 'max', default=85.0)
                    }
                }
            ),
            row=1, col=2
        )

    # Speed Analysis
    speed_stats = safe_get(analysis_data, 'speed', default={})
    if speed_stats:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=safe_get(speed_stats, 'mean', default=0.0),
                title={'text': "Speed (RPM)"},
                gauge={
                    'axis': {'range': [0, 4000]},
                    'steps': [
                        {'range': [0, 2800], 'color': "yellow"},
                        {'range': [2800, 3200], 'color': "green"},
                        {'range': [3200, 4000], 'color': "yellow"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': safe_get(speed_stats, 'max', default=3600.0)
                    }
                }
            ),
            row=2, col=1
        )

    # Spectrum Analysis if available
    spectrum_data = safe_get(analysis_data, 'vibration_spectrum', default={})
    if spectrum_data:
        frequencies = safe_get(spectrum_data, 'frequencies', default=[])
        psd = safe_get(spectrum_data, 'psd', default=[])
        if frequencies and psd:
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=psd,
                    name="Frequency Spectrum",
                    line=dict(color='blue')
                ),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
            fig.update_yaxes(title_text="Power", row=2, col=2)
    else:
        # Add placeholder text if no spectrum data
        fig.add_annotation(
            text="No spectrum data available",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        margin=dict(l=20, r=20, t=100, b=20),  # Increased top margin
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        annotations=[
            dict(
                text="Pattern Analysis Dashboard",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.2,  # Position above all subplots
                showarrow=False,
                font=dict(
                    size=28,
                    color="black",
                    family="Source Sans Pro"
                ),
                xanchor='center',
                yanchor='bottom'
            )
        ]
    )
    
    return fig

if __name__ == "__main__":
    main()