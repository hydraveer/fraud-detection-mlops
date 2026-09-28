"""Drift detection for fraud detection pipeline."""

from __future__ import annotations

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric


class DriftDetector:
    """Detects data drift between reference and current data."""

    def __init__(self, drift_threshold: float = 0.5) -> None:
        self.drift_threshold = drift_threshold

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        report_path: str = "reports/drift_report.html",
    ) -> dict:
        """Compare reference vs current data and return drift results."""

        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
        ])

        report.run(
            reference_data=reference_data,
            current_data=current_data
        )

        report.save_html(report_path)
        print(f"Drift report saved to {report_path}")

        results = report.as_dict()
        drift_detected = results["metrics"][1]["result"]["dataset_drift"]
        drift_share = results["metrics"][1]["result"]["share_of_drifted_columns"]

        return {
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "report_path": report_path,
        }
    