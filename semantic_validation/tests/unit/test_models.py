# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from semantic_validation.app.agent.models import Severity, SeverityType


class TestSeverityEvaluate:
    def test_high_severity(self):
        s = Severity(type=SeverityType.LOW)
        assert s.evaluate(0.0) == SeverityType.HIGH
        assert s.evaluate(0.4) == SeverityType.HIGH

    def test_medium_severity(self):
        s = Severity(type=SeverityType.LOW)
        assert s.evaluate(0.41) == SeverityType.MEDIUM
        assert s.evaluate(0.7) == SeverityType.MEDIUM

    def test_low_severity(self):
        s = Severity(type=SeverityType.LOW)
        assert s.evaluate(0.71) == SeverityType.LOW
        assert s.evaluate(1.0) == SeverityType.LOW

    def test_custom_thresholds(self):
        s = Severity(type=SeverityType.LOW, high=0.3, medium=0.6)
        assert s.evaluate(0.3) == SeverityType.HIGH
        assert s.evaluate(0.5) == SeverityType.MEDIUM
        assert s.evaluate(0.8) == SeverityType.LOW
