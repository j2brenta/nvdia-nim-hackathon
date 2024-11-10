# Automated Identification and Labeling of Industrial Time-Series Data

## How to Configure:
- Install Python 3.x on your machine.
- Clone or download this repository to your local machine.
- Open a terminal/command prompt and navigate to the project directory.
- Run `pip install -r requirements.txt` to install all required libraries.
- Create a `.env` file in the root directory and add your environment variables there. For example:
```
NVIDIA_API_KEY=your_api_key_here
Replace `your_api_key_here` with your actual API key for Nvidia.
```
- You're now ready to run the application!

## Running the Application:
To start the application, simply execute `streamlit run visualization.py`. This will launch the Flask web server which serves as our user interface.


# Project Information
This is a real-time monitoring system for industrial pumps that uses artificial intelligence to detect and predict potential failures.

### Primary Focus: Autonomous Knowledge Evolution
1. **System Components:**
   - Automatic pattern discovery
   - Self-validating models
   - Cross-process learning
   - Safety-bounded exploration

2. **Key Differentiators:**
   - No human in the loop for common cases
   - Automatic boundary detection
   - Self-expanding capabilities
   - Safety integrated


Here's how it works:

## Data Collection & Processing:
The system continuously collects three main sensor readings: temperature, vibration, and speed
The DataPipeline class processes this raw data through feature extraction and statistical analysis
Data is normalized and transformed into meaningful features for analysis

## Pattern Detection:
The PatternGenerator creates and recognizes four distinct operational patterns:
- Normal operation
- Bearing faults (characterized by specific vibration frequencies)
- Cavitation (showing turbulent flow patterns)
- Misalignment (displaying temperature increases and vibration)

The ModelEnsemble uses several AI models (multilayer perceptron, random forest, isolation forest) to detect these patterns reliably 
and calculates uncertainty in its predictions. I

## Safety Monitoring:
The SafetyValidator continuously checks if sensor readings are within safe operational limits
It can detect both immediate violations and concerning trends in data
Safety thresholds include maximum temperature (85°C), vibration (10 mm/s), and speed (3600 RPM)

## Learning System:
The system features a learning component that improves pattern recognition over time
It tracks metrics like detection speed, false positive rates, and recognition accuracy
The system continuously retrains its models and visualization shows a learning progress gauge and historical performance metrics.

Once the learning has identified a pattern and it is a safe condition, it is converted into vector form with Nvidia Embedding from NIM 
and stored into vector database. Then similar patterns are searched and resulting analysis is provided back if anomaly is detected.