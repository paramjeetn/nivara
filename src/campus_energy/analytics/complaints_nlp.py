"""NLP complaint clustering using TF-IDF and KMeans."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def _recommend_action(top_terms: list[str], theme_label: str = "") -> str:
    """Recommend corrective action based on cluster terms."""
    combined = f"{theme_label} {' '.join(top_terms)}".lower()

    if any(re.search(p, combined) for p in [r"\bhvac", r"\bcooling", r"\bac\b", r"\btemperature"]):
        return "Schedule preventive HVAC maintenance"
    if any(re.search(p, combined) for p in [r"\blight", r"\bbulb", r"\bflicker"]):
        return "Audit and replace ageing lighting fixtures"
    if any(re.search(p, combined) for p in [r"\bpump", r"\bwater", r"\bflow"]):
        return "Inspect water pumps and pipeline joints"
    if any(re.search(p, combined) for p in [r"\bsensor", r"\boffline", r"\bdata"]):
        return "Check sensor connectivity and BMS communication"
    return "Review facility maintenance log and escalate recurring complaints"


def cluster_complaints(complaints_df: pd.DataFrame) -> dict[str, Any]:
    """Cluster facility complaints using TF-IDF vectorization and KMeans.

    Args:
        complaints_df: DataFrame containing at least a 'description' column.

    Returns:
        dict containing:
            - 'cluster_df': DataFrame copy with 'cluster_id' and 'theme_label' columns
            - 'theme_counts': pd.Series of theme_label -> count, sorted descending
            - 'top_theme': str of most common theme label
            - 'top_theme_action': str of recommended action for top theme
            - 'tfidf_terms_per_cluster': dict of cluster_id -> list of top-5 terms
    """
    empty_result: dict[str, Any] = {
        "cluster_df": complaints_df.copy() if complaints_df is not None else pd.DataFrame(),
        "theme_counts": pd.Series(dtype=int),
        "top_theme": "",
        "top_theme_action": "",
        "tfidf_terms_per_cluster": {},
    }

    if complaints_df is None or len(complaints_df) < 3 or "description" not in complaints_df.columns:
        if isinstance(empty_result["cluster_df"], pd.DataFrame) and not empty_result["cluster_df"].empty:
            empty_result["cluster_df"]["cluster_id"] = None
            empty_result["cluster_df"]["theme_label"] = None
        return empty_result

    clean_text = complaints_df["description"].fillna("").astype(str)
    if clean_text.str.strip().replace("", np.nan).dropna().shape[0] < 3:
        return empty_result

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(clean_text)
    except ValueError:
        return empty_result

    feature_names = np.array(vectorizer.get_feature_names_out())
    if len(feature_names) == 0:
        return empty_result

    n_samples = len(complaints_df)
    n_clusters = min(6, n_samples // 2 or 1)
    n_clusters = max(1, min(n_clusters, n_samples))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    tfidf_terms_per_cluster: dict[int, list[str]] = {}
    cluster_theme_map: dict[int, str] = {}

    for cluster_id in range(n_clusters):
        top_k = min(5, len(feature_names))
        top_indices = order_centroids[cluster_id, :top_k]
        top_terms = [str(feature_names[idx]) for idx in top_indices]
        tfidf_terms_per_cluster[cluster_id] = top_terms

        # Format human-readable theme label from top terms
        label_terms = [t.title() for t in top_terms[:3]]
        cluster_theme_map[cluster_id] = " / ".join(label_terms) if label_terms else f"Cluster {cluster_id + 1}"

    result_df = complaints_df.copy()
    result_df["cluster_id"] = labels
    result_df["theme_label"] = result_df["cluster_id"].map(cluster_theme_map)

    theme_counts = result_df["theme_label"].value_counts()
    top_theme = str(theme_counts.index[0]) if not theme_counts.empty else ""

    top_cluster_id = result_df.loc[result_df["theme_label"] == top_theme, "cluster_id"].iloc[0] if top_theme else 0
    top_terms = tfidf_terms_per_cluster.get(int(top_cluster_id), [])
    top_theme_action = _recommend_action(top_terms, top_theme) if top_theme else ""

    return {
        "cluster_df": result_df,
        "theme_counts": theme_counts,
        "top_theme": top_theme,
        "top_theme_action": top_theme_action,
        "tfidf_terms_per_cluster": tfidf_terms_per_cluster,
    }
