# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum


class DistillStatus(StrEnum):
    DISTILLED   = "distilled"    # existing relation was processed in a distillation batch
    SYNTHESIZED = "synthesized"  # relation was created by CoDi (anchor → CoDiN summary link)
    PRUNED      = "pruned"       # removed from active graph
