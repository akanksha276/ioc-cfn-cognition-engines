# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Mock data repository implementation.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MockDataRepository:
    """Mock implementation of the data repository."""

    def load_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load OTEL data from a JSON or JSONL file.
        
        Args:
            file_path: Path to the JSON/JSONL file
            
        Returns:
            List of OTEL records
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix not in ['.json', '.jsonl']:
            raise ValueError("File must be a JSON or JSONL file")

        with open(path, 'r') as f:
            if path.suffix == '.jsonl':
                # JSONL: each line is a separate JSON object
                return [json.loads(line.strip()) for line in f if line.strip()]
            else:
                # Regular JSON
                data = json.load(f)
                return data if isinstance(data, list) else [data]

    def parse_body(self, body: bytes) -> List[Dict[str, Any]]:
        """
        Parse OTEL data from request body (JSON array or NDJSON).
        
        Args:
            body: Raw request body bytes
            
        Returns:
            List of OTEL records
            
        Raises:
            ValueError: If no valid records found
        """
        body_str = body.decode('utf-8').strip()

        otel_data = []
        if body_str.startswith('['):
            otel_data = json.loads(body_str)
        else:
            # NDJSON format
            for line in body_str.split('\n'):
                if line.strip():
                    otel_data.append(json.loads(line.strip()))

        if not otel_data:
            raise ValueError("No valid OTEL records found in request body")

        return otel_data

    def save_output(self, data: Dict[str, Any], filename: str) -> bool:
        """
        Save extraction output to a JSON file.
        
        Args:
            data: Extraction result data
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, "w") as outfile:
                json.dump(data, outfile, indent=2)
            logger.info(f"Saved extraction result to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save extraction result to {filename}: {e}")
            return False

