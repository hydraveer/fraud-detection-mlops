import pandas as pd
import numpy as np
from src.drift_detector import DriftDetector

# Reference data (January - what model trained on)
np.random.seed(42)
reference = pd.DataFrame({
    "Amount": np.random.normal(100, 50, 1000),
    "V1": np.random.normal(0, 1, 1000),
    "V2": np.random.normal(0, 1, 1000),
    "Class": np.random.choice([0, 1], 1000, p=[0.998, 0.002])
})

# Current data (July - new fraud patterns)
current = pd.DataFrame({
    "Amount": np.random.normal(500, 200, 500),  # amounts shifted up
    "V1": np.random.normal(2, 1, 500),           # feature distribution changed
    "V2": np.random.normal(-2, 1, 500),          # feature distribution changed
    "Class": np.random.choice([0, 1], 500, p=[0.99, 0.01])
})

detector = DriftDetector()
results = detector.detect(reference, current)

print(f"Drift detected: {results['drift_detected']}")
print(f"Drift share: {results['drift_share']:.2%}")

if results['drift_detected']:
    print("⚠️ Model needs retraining!")
else:
    print("✅ No drift detected")