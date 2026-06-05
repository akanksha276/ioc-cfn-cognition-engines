# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Data repository protocol/interface for data access abstraction.
"""
from pathlib import Path
from typing import Any, Dict, List, Protocol


class DataRepository(Protocol):
    """Protocol for data repository implementations."""

    def load_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load OTEL data from a file.
        
        Args:
            file_path: Path to the JSON/JSONL file
            
        Returns:
            List of OTEL records
        """
        ...

    def parse_body(self, body: bytes) -> List[Dict[str, Any]]:
        """
        Parse OTEL data from request body.
        
        Args:
            body: Raw request body bytes
            
        Returns:
            List of OTEL records
        """
        ...

    def save_output(self, data: Dict[str, Any], filename: str) -> bool:
        """
        Save extraction output to a file.
        
        Args:
            data: Extraction result data
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        ...

