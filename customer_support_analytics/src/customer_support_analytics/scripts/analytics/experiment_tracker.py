"""
Experiment Tracking Framework for SupportIQ

This module provides comprehensive experiment management capabilities:
- Experiment metadata tracking
- Parameter configuration management
- Metric logging and versioning
- Experiment comparison and selection
- Results caching and reproducibility

Use this module to manage multiple experiments systematically,
enabling reproducible research and informed decision-making.

Reference: https://vinted.com - Support Agent Tooling team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import copy

from supportiq_web import logger, path_manager


class ExperimentTracker:
    """
    Experiment Tracking and Management System.

    Provides capabilities for:
    - Creating and managing experiments
    - Logging parameters and metrics
    - Tracking experiment lineage
    - Comparing experiment results
    - Reproducing past experiments
    """

    def __init__(self, experiment_dir: Path = None):
        """
        Initialize Experiment Tracker.

        Args:
            experiment_dir: Directory for storing experiment data
        """
        if experiment_dir is None:
            experiment_dir = path_manager.DATA_DIR / 'experiments'

        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.experiments_file = self.experiment_dir / 'experiments_registry.json'
        self.experiments = self._load_experiments()

    def _load_experiments(self) -> Dict:
        """Load experiments registry from disk."""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return {'experiments': [], 'version': '1.0'}

    def _save_experiments(self):
        """Save experiments registry to disk."""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def create_experiment(
        self,
        name: str,
        description: str = "",
        hypothesis: str = "",
        tags: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> str:
        """
        Create a new experiment entry.

        Args:
            name: Experiment name
            description: Experiment description
            hypothesis: Research hypothesis
            tags: List of tags for categorization
            parameters: Experiment parameters/configuration

        Returns:
            Experiment ID
        """
        experiment_id = self._generate_experiment_id(name, parameters or {})

        experiment = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'hypothesis': hypothesis,
            'tags': tags or [],
            'parameters': parameters or {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'status': 'created',
            'metrics': {},
            'results': {},
            'notes': [],
            'artifacts': []
        }

        self.experiments['experiments'].append(experiment)
        self._save_experiments()

        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment_id

    def _generate_experiment_id(self, name: str, parameters: Dict) -> str:
        """Generate unique experiment ID based on name and parameters."""
        content = f"{name}_{json.dumps(parameters, sort_keys=True)}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        metric_value: float,
        step: Optional[int] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Log a metric value for an experiment.

        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            metric_value: Value of the metric
            step: Optional step/epoch number
            timestamp: Optional timestamp (defaults to now)

        Returns:
            True if successful, False otherwise
        """
        experiment = self._find_experiment(experiment_id)
        if experiment is None:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        if metric_name not in experiment['metrics']:
            experiment['metrics'][metric_name] = []

        metric_entry = {
            'value': metric_value,
            'timestamp': timestamp or datetime.now().isoformat(),
            'step': step
        }

        experiment['metrics'][metric_name].append(metric_entry)
        experiment['updated_at'] = datetime.now().isoformat()
        self._save_experiments()

        logger.debug(f"Logged metric {metric_name}={metric_value} for experiment {experiment_id}")
        return True

    def log_parameters(
        self,
        experiment_id: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Update parameters for an experiment.

        Args:
            experiment_id: Experiment ID
            parameters: Parameters to log

        Returns:
            True if successful, False otherwise
        """
        experiment = self._find_experiment(experiment_id)
        if experiment is None:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        experiment['parameters'].update(parameters)
        experiment['updated_at'] = datetime.now().isoformat()
        self._save_experiments()

        return True

    def log_results(
        self,
        experiment_id: str,
        results: Dict[str, Any]
    ) -> bool:
        """
        Log final results for an experiment.

        Args:
            experiment_id: Experiment ID
            results: Results dictionary

        Returns:
            True if successful, False otherwise
        """
        experiment = self._find_experiment(experiment_id)
        if experiment is None:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        experiment['results'] = results
        experiment['updated_at'] = datetime.now().isoformat()
        self._save_experiments()

        logger.info(f"Logged results for experiment {experiment_id}")
        return True

    def add_note(
        self,
        experiment_id: str,
        note: str,
        note_type: str = 'general'
    ) -> bool:
        """
        Add a note to an experiment.

        Args:
            experiment_id: Experiment ID
            note: Note content
            note_type: Type of note ('general', 'observation', 'issue', 'conclusion')

        Returns:
            True if successful, False otherwise
        """
        experiment = self._find_experiment(experiment_id)
        if experiment is None:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False

        note_entry = {
            'content': note,
            'type': note_type,
            'timestamp': datetime.now().isoformat()
        }

        experiment['notes'].append(note_entry)
        experiment['updated_at'] = datetime.now().isoformat()
        self._save_experiments()

        return True

    def update_status(
        self,
        experiment_id: str,
        status: str
    ) -> bool:
        """
        Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status ('created', 'running', 'completed', 'failed', 'cancelled')

        Returns:
            True if successful, False otherwise
        """
        valid_statuses = ['created', 'running', 'completed', 'failed', 'cancelled']
        if status not in valid_statuses:
            logger.warning(f"Invalid status: {status}")
            return False

        experiment = self._find_experiment(experiment_id)
        if experiment is None:
            return False

        experiment['status'] = status
        experiment['updated_at'] = datetime.now().isoformat()
        self._save_experiments()

        logger.info(f"Updated experiment {experiment_id} status to {status}")
        return True

    def _find_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Find experiment by ID."""
        for exp in self.experiments['experiments']:
            if exp['id'] == experiment_id:
                return exp
        return None

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        Get experiment details.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment dictionary or None
        """
        return copy.deepcopy(self._find_experiment(experiment_id))

    def list_experiments(
        self,
        tags: List[str] = None,
        status: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        List experiments with optional filtering.

        Args:
            tags: Filter by tags
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of experiment dictionaries
        """
        results = []

        for exp in self.experiments['experiments']:
            # Apply tag filter
            if tags:
                if not any(tag in exp['tags'] for tag in tags):
                    continue

            # Apply status filter
            if status and exp['status'] != status:
                continue

            results.append(copy.deepcopy(exp))

        # Sort by updated_at descending
        results.sort(key=lambda x: x['updated_at'], reverse=True)

        return results[:limit]

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str] = None
    ) -> Dict:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metric names to compare (None = all common metrics)

        Returns:
            Comparison results dictionary
        """
        experiments = []
        for exp_id in experiment_ids:
            exp = self._find_experiment(exp_id)
            if exp:
                experiments.append(exp)

        if len(experiments) < 2:
            return {'error': 'Need at least 2 experiments to compare'}

        # Find common metrics
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp['metrics'].keys())

        if metrics:
            comparison_metrics = [m for m in metrics if m in all_metrics]
        else:
            comparison_metrics = list(all_metrics)

        # Build comparison table
        comparison = {
            'experiments': [],
            'metrics': {},
            'best_per_metric': {}
        }

        for exp in experiments:
            exp_summary = {
                'id': exp['id'],
                'name': exp['name'],
                'status': exp['status'],
                'tags': exp['tags']
            }
            comparison['experiments'].append(exp_summary)

        # Compare metrics
        for metric in comparison_metrics:
            metric_values = {}
            for exp in experiments:
                if metric in exp['metrics'] and exp['metrics'][metric]:
                    # Get latest value
                    latest = exp['metrics'][metric][-1]
                    metric_values[exp['id']] = latest['value']

            if metric_values:
                comparison['metrics'][metric] = metric_values

                # Find best value (highest for metrics like accuracy, lowest for metrics like loss)
                best_id = max(metric_values, key=metric_values.get)
                comparison['best_per_metric'][metric] = {
                    'experiment_id': best_id,
                    'value': metric_values[best_id]
                }

        return comparison

    def get_metric_summary(
        self,
        experiment_id: str,
        metric_name: str
    ) -> Optional[Dict]:
        """
        Get summary statistics for a metric across all logged values.

        Args:
            experiment_id: Experiment ID
            metric_name: Metric name

        Returns:
            Summary dictionary with mean, std, min, max, etc.
        """
        experiment = self._find_experiment(experiment_id)
        if experiment is None:
            return None

        if metric_name not in experiment['metrics']:
            return None

        values = [m['value'] for m in experiment['metrics'][metric_name]]

        if not values:
            return None

        return {
            'metric': metric_name,
            'count': len(values),
            'mean': round(np.mean(values), 6),
            'std': round(np.std(values), 4),
            'min': round(np.min(values), 6),
            'max': round(np.max(values), 6),
            'first': values[0],
            'last': values[-1],
            'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable'
        }

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if successful, False otherwise
        """
        for i, exp in enumerate(self.experiments['experiments']):
            if exp['id'] == experiment_id:
                del self.experiments['experiments'][i]
                self._save_experiments()
                logger.info(f"Deleted experiment {experiment_id}")
                return True

        return False

    def get_best_experiment(
        self,
        metric: str,
        tags: List[str] = None,
        minimize: bool = False
    ) -> Optional[Dict]:
        """
        Get the best experiment based on a metric.

        Args:
            metric: Metric to optimize
            tags: Optional tag filter
            minimize: If True, lower is better; if False, higher is better

        Returns:
            Best experiment dictionary or None
        """
        experiments = self.list_experiments(tags=tags, status='completed')

        best_exp = None
        best_value = float('inf') if minimize else float('-inf')

        for exp in experiments:
            if metric in exp['metrics'] and exp['metrics'][metric]:
                value = exp['metrics'][metric][-1]['value']

                if minimize:
                    if value < best_value:
                        best_value = value
                        best_exp = exp
                else:
                    if value > best_value:
                        best_value = value
                        best_exp = exp

        return copy.deepcopy(best_exp) if best_exp else None

    def export_experiment(
        self,
        experiment_id: str,
        format: str = 'json'
    ) -> Optional[str]:
        """
        Export experiment data.

        Args:
            experiment_id: Experiment ID
            format: Export format ('json', 'dict')

        Returns:
            Exported data as string or None
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return None

        if format == 'json':
            return json.dumps(experiment, indent=2)
        elif format == 'dict':
            return experiment

        return str(experiment)


class ExperimentRunner:
    """
    Context manager for running experiments with automatic tracking.

    Usage:
        with ExperimentRunner(tracker, "my_experiment") as exp:
            exp.log_metric("accuracy", 0.95)
            exp.log_results({"final_accuracy": 0.95})
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        name: str,
        description: str = "",
        parameters: Dict = None,
        tags: List[str] = None
    ):
        """
        Initialize Experiment Runner.

        Args:
            tracker: ExperimentTracker instance
            name: Experiment name
            description: Experiment description
            parameters: Initial parameters
            tags: Tags for the experiment
        """
        self.tracker = tracker
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.tags = tags or []
        self.experiment_id = None

    def __enter__(self):
        """Start the experiment."""
        self.experiment_id = self.tracker.create_experiment(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            tags=self.tags
        )
        self.tracker.update_status(self.experiment_id, 'running')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the experiment."""
        if exc_type is not None:
            # Experiment failed
            self.tracker.add_note(
                self.experiment_id,
                f"Experiment failed with error: {exc_val}",
                note_type='issue'
            )
            self.tracker.update_status(self.experiment_id, 'failed')
        else:
            # Experiment completed successfully
            self.tracker.update_status(self.experiment_id, 'completed')

        return False  # Don't suppress exceptions

    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric value."""
        self.tracker.log_metric(self.experiment_id, name, value, step)

    def log_parameters(self, parameters: Dict):
        """Log additional parameters."""
        self.tracker.log_parameters(self.experiment_id, parameters)

    def log_results(self, results: Dict):
        """Log final results."""
        self.tracker.log_results(self.experiment_id, results)

    def add_note(self, note: str, note_type: str = 'general'):
        """Add a note."""
        self.tracker.add_note(self.experiment_id, note, note_type)


# Convenience functions
def create_default_tracker() -> ExperimentTracker:
    """Create a default experiment tracker."""
    return ExperimentTracker()


def quick_experiment(
    tracker: ExperimentTracker,
    name: str,
    parameters: Dict,
    run_fn: callable
) -> Dict:
    """
    Run a quick experiment with automatic tracking.

    Args:
        tracker: ExperimentTracker instance
        name: Experiment name
        parameters: Experiment parameters
        run_fn: Function that takes parameters and returns results dict

    Returns:
        Results dictionary
    """
    with ExperimentRunner(tracker, name, parameters=parameters) as exp:
        results = run_fn(**parameters)
        exp.log_results(results)
        return results
