# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Base SDK for Knowledge Extraction and Ingestion Platform (KXP) Adapters.
Provides common functionality for data source adapters.
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MetricsObject:
    """Operational metrics for the adapter."""
    records_processed: int = 0
    records_sent: int = 0
    records_failed: int = 0
    last_run_timestamp: Optional[datetime] = None
    last_run_duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class AdapterConfig:
    """Configuration for adapter initialization."""
    data_source_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    extraction_rules: Dict[str, Any]


class AdapterSDK:
    """
    Base SDK class for KXP adapters.
    Provides common functionality for data extraction.
    """

    def __init__(self):
        self.config: Optional[AdapterConfig] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._default_log_level = logging.INFO
        self.logger.setLevel(self._default_log_level)
        self.metrics = MetricsObject()
        self._initialized = False

    def load(self) -> Dict[str, Any]:
        """
        Load data from the configured data source.
        
        Returns:
            Dict containing loaded data and status
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call init() first.")

        start_time = time.time()
        try:
            self.logger.info("Loading data from source...")
            result = self._load_impl()

            duration = time.time() - start_time
            self.metrics.last_run_timestamp = datetime.now()
            self.metrics.last_run_duration_seconds = duration

            self.logger.info(f"Data loaded successfully in {duration:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            self.metrics.errors.append(str(e))
            raise

    def _load_impl(self) -> Dict[str, Any]:
        """Implementation of load logic - to be overridden by specific adapters."""
        raise NotImplementedError("Subclasses must implement _load_impl()")

    def report_health_and_diagnostics(self) -> Dict[str, Any]:
        """
        Report health status and diagnostic information.
        
        Returns:
            Dict containing health status and diagnostics
        """
        health_info = {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "metrics": {
                "records_processed": self.metrics.records_processed,
                "records_sent": self.metrics.records_sent,
                "records_failed": self.metrics.records_failed,
                "last_run": self.metrics.last_run_timestamp.isoformat() if self.metrics.last_run_timestamp else None,
                "last_run_duration_seconds": self.metrics.last_run_duration_seconds
            },
            "recent_errors": self.metrics.errors[-5:]
        }
        return health_info

    def set_log_level(self, level: int):
        """
        Set the logging level for the adapter.
        
        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO, etc.)
        """
        self.logger.setLevel(level)
        self.logger.info(f"Log level set to: {logging.getLevelName(level)}")

    def reset_log_level(self):
        """Reset logging level to default (INFO)."""
        self.logger.setLevel(self._default_log_level)
        self.logger.info(f"Log level reset to: {logging.getLevelName(self._default_log_level)}")

    def get_operational_metrics(self) -> MetricsObject:
        """
        Get operational metrics for the adapter.
        
        Returns:
            MetricsObject with current metrics
        """
        return self.metrics

