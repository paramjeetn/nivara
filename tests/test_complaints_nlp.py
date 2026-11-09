"""Tests for complaint clustering NLP module."""

import pandas as pd
from campus_energy.analytics.complaints_nlp import cluster_complaints


def test_cluster_complaints_empty():
    res = cluster_complaints(pd.DataFrame())
    assert res["top_theme"] == ""
    assert res["top_theme_action"] == ""
    assert res["theme_counts"].empty
    assert res["tfidf_terms_per_cluster"] == {}


def test_cluster_complaints_too_few_rows():
    df = pd.DataFrame({"description": ["AC is broken.", "Light is flickering."]})
    res = cluster_complaints(df)
    assert res["top_theme"] == ""
    assert res["top_theme_action"] == ""


def test_cluster_complaints_generated_data():
    df = pd.read_csv("data/generated/complaints.csv")
    res = cluster_complaints(df)
    assert not res["theme_counts"].empty
    assert len(res["top_theme"]) > 0
    assert len(res["top_theme_action"]) > 0
    assert "cluster_id" in res["cluster_df"].columns
    assert "theme_label" in res["cluster_df"].columns
    assert len(res["tfidf_terms_per_cluster"]) > 0
    # Top theme action should be a non-empty recommended action
    assert any(
        kw in res["top_theme_action"]
        for kw in ["maintenance", "fixtures", "joints", "connectivity", "Review facility"]
    )
