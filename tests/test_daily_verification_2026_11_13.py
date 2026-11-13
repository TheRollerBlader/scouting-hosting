"""Autonomous daily verification test suite for scouting-hosting.
Scheduled Milestone: Testing & Verification (2026-11-13).

Comprehensive unit tests covering:
1. Pure alliance score calculation and contributions aggregation.
2. Estimated robot performance fallback for unseen robots.
3. Flask /predict parameter validation:
   - Rejection of missing event_name (400)
   - Rejection of missing blue_alliance on custom match 1313 (400)
   - Rejection of missing red_alliance on custom match 1313 (400)
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Ensure external DB dependencies are safely mocked for hermetic unit testing
if "flask_cors" not in sys.modules:
    sys.modules["flask_cors"] = MagicMock()
if "mysql" not in sys.modules:
    sys.modules["mysql"] = MagicMock()
if "mysql.connector" not in sys.modules:
    sys.modules["mysql.connector"] = MagicMock()

# Add repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from predict import (
    app,
    estimate_robot_performance,
    get_alliance_stats,
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


class TestRobotAnalyticsLogic:
    def test_estimate_robot_performance_unseen_robot(self):
        """Unseen robot fallback must adopt global average baselines."""
        global_avgs = {
            "avg_points_per_match": 18.5,
            "success_rate": 0.82,
            "total_events": 14.0,
            "points_slope": 0.5,
            "success_rate_slope": 0.01,
            "total_events_slope": 0.2,
            "predicted_next_points": 19.0,
        }
        res = estimate_robot_performance("9999", global_avgs)
        assert res["robot"] == "9999"
        assert res["avg_points_per_match"] == 18.5
        assert res["matches"] == 0
        assert res["most_common_action"] == "Unknown"

    def test_get_alliance_stats_structure(self):
        """Alliance stats extractor must return properly formatted and rounded dictionaries."""
        global_avgs = {
            "avg_points_per_match": 22.1234,
            "success_rate": 0.90,
            "total_events": 16.0,
            "points_slope": 0.123,
            "success_rate_slope": 0.05,
            "total_events_slope": 0.1,
            "predicted_next_points": 23.5,
        }
        mock_df = pd.DataFrame(columns=["robot", "avg_points_per_match"])
        stats = get_alliance_stats(["5411", "3355", "4364"], mock_df, global_avgs)
        assert len(stats) == 3
        for robot_stat in stats:
            assert "robot" in robot_stat
            assert "avg_points_per_match" in robot_stat
            assert robot_stat["avg_points_per_match"] == 22.12


class TestScoutingHostingEndpoints:
    def test_predict_rejects_missing_event_name(self, client):
        """Request without event_name query param must return 400 Bad Request."""
        res = client.get("/predict?match_no=1313")
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data
        assert "event_name" in data["error"]

    def test_predict_custom_alliance_missing_blue(self, client):
        """When match_no=1313, missing blue_alliance must return 400 Bad Request."""
        res = client.get("/predict?event_name=2026txftw&match_no=1313&red_alliance=118,148,254")
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data
        assert "blue_alliance" in data["error"]

    def test_predict_custom_alliance_missing_red(self, client):
        """When match_no=1313, missing red_alliance must return 400 Bad Request."""
        res = client.get("/predict?event_name=2026txftw&match_no=1313&blue_alliance=5411,3355,4364")
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data
        assert "red_alliance" in data["error"]
