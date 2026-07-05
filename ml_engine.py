"""
ml_engine.py
------------
Lightweight, explainable ML layer on top of the route summary produced by
data_engine.py:

    1. cluster_routes()      -> KMeans tiers: Efficient / Moderate / Bottleneck
    2. detect_geo_bottlenecks() -> high-volume + poor-performance states
    3. lead_time_drivers()   -> RandomForest feature importance
                                 (which factors move lead time the most)

Kept deliberately simple / interpretable — the goal is decision support for
a logistics stakeholder, not a black-box predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


TIER_LABELS = ["Bottleneck", "Moderate", "Efficient"]


def cluster_routes(route_summary: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Groups routes into performance tiers using KMeans over
    (avg lead time, lead time variability, delay frequency).
    Clusters are then ranked by mean lead time so labels are consistent
    (lowest lead time = 'Efficient') regardless of KMeans' arbitrary
    cluster ordering."""

    df = route_summary.copy()
    if len(df) < n_clusters:
        df["Performance_Tier"] = "Efficient"
        return df

    features = df[["Avg_Lead_Time", "Lead_Time_StdDev", "Delay_Frequency_Pct"]].fillna(0)
    scaled = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(scaled)
    df["_cluster"] = raw_labels

    # rank clusters by mean lead time -> map to human-readable tiers
    cluster_order = (
        df.groupby("_cluster")["Avg_Lead_Time"].mean().sort_values().index.tolist()
    )
    n = len(cluster_order)
    labels = TIER_LABELS if n == 3 else [f"Tier {i+1}" for i in range(n)]
    tier_map = {cluster_id: labels[i] for i, cluster_id in enumerate(cluster_order)}
    df["Performance_Tier"] = df["_cluster"].map(tier_map)
    df = df.drop(columns=["_cluster"])
    return df


def detect_geo_bottlenecks(geo_summary: pd.DataFrame, volume_pctile: float = 0.5,
                             lead_pctile: float = 0.5) -> pd.DataFrame:
    """Flags states that are simultaneously high-volume (>= volume_pctile)
    and slow (>= lead_pctile average lead time) -- i.e. congestion-prone
    rather than just occasionally slow on a low-volume lane."""

    df = geo_summary.copy()
    vol_cut = df["Shipment_Volume"].quantile(volume_pctile)
    lead_cut = df["Avg_Lead_Time"].quantile(lead_pctile)

    df["Is_Bottleneck"] = (df["Shipment_Volume"] >= vol_cut) & (df["Avg_Lead_Time"] >= lead_cut)
    return df.sort_values(["Is_Bottleneck", "Avg_Lead_Time"], ascending=[False, False])


def lead_time_drivers(df: pd.DataFrame, sample_cap: int = 8000, random_state: int = 42) -> pd.DataFrame:
    """Fits a small RandomForestRegressor on Ship Mode / Region / Factory /
    Division to explain what drives shipping lead time, and returns a
    feature-importance table for the dashboard. This is descriptive
    (explains variance already in the data), not a forecast."""

    work = df[["Ship Mode", "Region", "Factory", "Division", "Lead Time"]].dropna().copy()
    if len(work) > sample_cap:
        work = work.sample(sample_cap, random_state=random_state)

    cat_cols = ["Ship Mode", "Region", "Factory", "Division"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(work[cat_cols])
    feature_names = encoder.get_feature_names_out(cat_cols)

    y = work["Lead Time"].values

    model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state, n_jobs=-1)
    model.fit(X_cat, y)

    # roll per-category importance back up to the parent field
    importances = pd.Series(model.feature_importances_, index=feature_names)
    rollup = {}
    for col in cat_cols:
        rollup[col] = importances[[f for f in feature_names if f.startswith(col + "_")]].sum()

    result = pd.DataFrame({
        "Factor": list(rollup.keys()),
        "Relative Importance": list(rollup.values()),
    }).sort_values("Relative Importance", ascending=False)
    result["Relative Importance"] = (result["Relative Importance"] / result["Relative Importance"].sum() * 100).round(1)
    return result
