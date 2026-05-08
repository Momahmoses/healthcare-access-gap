"""
Healthcare access gap analysis: coverage scoring, gap identification,
and optimal facility placement recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

METRICS_PATH = Path("assets/metrics.json")
GAP_FEATURES = [
    "population", "is_rural", "road_quality", "nearest_facility_km",
    "n_facilities_10km", "doctors_per_1000", "travel_time_min", "health_insurance_pct",
]


def identify_gap_clusters(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Cluster underserved communities using K-Means to identify geographic gap zones.

    Parameters
    ----------
    df : pd.DataFrame
        Community dataset with GAP_FEATURES and lat/lon.
    n_clusters : int
        Number of gap cluster zones to identify.

    Returns
    -------
    pd.DataFrame
        Input dataframe with added cluster_id and cluster_priority columns.
    """
    underserved = df[df["access_tier"] == "Underserved"].copy()
    X = StandardScaler().fit_transform(underserved[["latitude", "longitude", "population", "travel_time_min"]])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    underserved["cluster_id"] = km.fit_predict(X)

    cluster_stats = (
        underserved.groupby("cluster_id")
        .agg(total_population=("population", "sum"), mean_travel_min=("travel_time_min", "mean"),
             n_communities=("community_id", "count"), centroid_lat=("latitude", "mean"),
             centroid_lon=("longitude", "mean"))
        .reset_index()
        .sort_values("total_population", ascending=False)
    )
    cluster_stats["priority_rank"] = range(1, len(cluster_stats) + 1)
    return underserved.merge(cluster_stats[["cluster_id", "priority_rank"]], on="cluster_id"), cluster_stats


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    total_pop = df["population"].sum()
    underserved_pop = df[df["access_tier"] == "Underserved"]["population"].sum()
    return {
        "total_communities": total,
        "total_population": int(total_pop),
        "underserved_communities": int((df["access_tier"] == "Underserved").sum()),
        "underserved_population": int(underserved_pop),
        "underserved_population_pct": round(underserved_pop / total_pop * 100, 2),
        "mean_travel_time_min": round(df["travel_time_min"].mean(), 1),
        "mean_doctors_per_1000": round(df["doctors_per_1000"].mean(), 3),
        "rural_underserved_pct": round(
            df[(df["is_rural"] == 1) & (df["access_tier"] == "Underserved")]["population"].sum()
            / df[df["is_rural"] == 1]["population"].sum() * 100, 2
        ),
    }


def save_metrics(metrics: dict) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
