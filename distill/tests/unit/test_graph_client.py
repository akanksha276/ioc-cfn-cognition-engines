# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from distill.app.data.graph_client import graph_update_post_url


def test_graph_update_post_url_matches_cfn_internal_route():
    u = graph_update_post_url("https://api.example", "ws-1", "mas-2", "update")
    assert u == (
        "https://api.example/api/internal/workspaces/ws-1/multi-agentic-systems/mas-2/graph/update"
    )
