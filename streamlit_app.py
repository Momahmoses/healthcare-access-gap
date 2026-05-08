"""
Healthcare Facility Access Gap Analysis
========================================
Identifies underserved Nigerian communities using travel time, facility density,
workforce metrics, and population data. Recommends priority sites for new clinics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import generate_community_dataset
from model import compute_summary_metrics, identify_gap_clusters, save_metrics, METRICS_PATH

st.set_page_config(
    page_title="Healthcare Access Gap | Nigeria",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIER_COLORS = {"Well-served": "#2ECC71", "Moderate": "#F39C12", "Underserved": "#E74C3C"}


@st.cache_resource(show_spinner="Analyzing healthcare access gaps…")
def load_data():
    df = generate_community_dataset()
    metrics = compute_summary_metrics(df)
    save_metrics(metrics)
    underserved_df, cluster_stats = identify_gap_clusters(df)
    return df, metrics, underserved_df, cluster_stats


df, metrics, underserved_df, cluster_stats = load_data()

with st.sidebar:
    st.title("🏥 Healthcare Access")
    st.caption("Nigeria Gap Analysis Platform")
    st.divider()
    page = st.radio("Navigation", ["Overview", "Access Map", "Gap Zones", "State Analysis"], label_visibility="collapsed")
    st.divider()
    state_filter = st.multiselect("Filter by State", sorted(df["state"].unique()), default=sorted(df["state"].unique()))
    tier_filter = st.multiselect("Filter by Access Tier", ["Well-served", "Moderate", "Underserved"], default=["Well-served", "Moderate", "Underserved"])

df_f = df[(df["state"].isin(state_filter)) & (df["access_tier"].isin(tier_filter))]

if page == "Overview":
    st.title("Healthcare Facility Access Gap Analysis")
    st.markdown("Identifying underserved communities and optimal placement zones for new health facilities across North-Central and Northeast Nigeria.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Communities Assessed", f"{metrics['total_communities']:,}")
    c2.metric("Underserved Population", f"{metrics['underserved_population']:,}", f"{metrics['underserved_population_pct']}% of total", delta_color="inverse")
    c3.metric("Mean Travel Time to Facility", f"{metrics['mean_travel_time_min']:.0f} min")
    c4.metric("Mean Doctors per 1,000", f"{metrics['mean_doctors_per_1000']:.3f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        tier_counts = df_f["access_tier"].value_counts().reset_index()
        fig = px.pie(tier_counts, values="count", names="access_tier",
                     color="access_tier", color_discrete_map=TIER_COLORS,
                     title="Community Distribution by Access Tier", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df_f, x="travel_time_min", color="access_tier",
                            color_discrete_map=TIER_COLORS, nbins=50,
                            title="Travel Time Distribution to Nearest Facility",
                            labels={"travel_time_min": "Travel Time (minutes)"}, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df_f.sample(min(2000, len(df_f)), random_state=42),
                          x="nearest_facility_km", y="travel_time_min", color="access_tier",
                          color_discrete_map=TIER_COLORS, opacity=0.5,
                          title="Distance vs Travel Time by Access Tier",
                          labels={"nearest_facility_km": "Nearest Facility (km)", "travel_time_min": "Travel Time (min)"}, height=380)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        rural_urban = df_f.groupby(["access_tier", df_f["is_rural"].map({0: "Urban", 1: "Rural"})])["population"].sum().reset_index()
        rural_urban.columns = ["access_tier", "area_type", "population"]
        fig4 = px.bar(rural_urban, x="access_tier", y="population", color="area_type",
                      title="Population by Access Tier and Area Type", barmode="group",
                      color_discrete_sequence=["#3498DB", "#95A5A6"], height=380)
        st.plotly_chart(fig4, use_container_width=True)

elif page == "Access Map":
    st.title("Geographic Access Coverage Map")
    sample = df_f.sample(min(1500, len(df_f)), random_state=42)
    fig = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude",
        color="access_tier", color_discrete_map=TIER_COLORS,
        size="population", size_max=15,
        hover_name="state",
        hover_data={"access_tier": True, "travel_time_min": ":.0f", "population": ":,", "latitude": False, "longitude": False},
        mapbox_style="carto-positron", zoom=5,
        center={"lat": 9.0, "lon": 8.5},
        title="Community Healthcare Access Tiers — Nigeria", height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Gap Zones":
    st.title("Priority Gap Zones for New Facilities")
    st.markdown("K-Means clustering of underserved communities to identify the highest-priority zones for new clinic or hospital placement.")
    st.divider()

    st.subheader("Top Priority Zones by Affected Population")
    display_cols = ["priority_rank", "total_population", "mean_travel_min", "n_communities", "centroid_lat", "centroid_lon"]
    st.dataframe(
        cluster_stats.sort_values("priority_rank")[display_cols].rename(columns={
            "priority_rank": "Priority", "total_population": "Affected Population",
            "mean_travel_min": "Avg Travel (min)", "n_communities": "Communities",
            "centroid_lat": "Center Lat", "centroid_lon": "Center Lon",
        }).reset_index(drop=True),
        use_container_width=True,
    )

    fig = px.scatter_mapbox(
        underserved_df, lat="latitude", lon="longitude",
        color="cluster_id", size="population", size_max=12,
        hover_name="state",
        hover_data={"access_tier": True, "travel_time_min": ":.0f", "priority_rank": True, "latitude": False, "longitude": False},
        mapbox_style="carto-positron", zoom=5,
        center={"lat": 9.0, "lon": 8.5},
        title="Underserved Community Clusters (Numbered by Priority)", height=520,
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "State Analysis":
    st.title("State-Level Healthcare Access Analysis")
    state_summary = (
        df_f.groupby("state").agg(
            total_communities=("community_id", "count"),
            underserved=("access_tier", lambda x: (x == "Underserved").sum()),
            mean_travel=("travel_time_min", "mean"),
            mean_doctors=("doctors_per_1000", "mean"),
            total_population=("population", "sum"),
        ).reset_index()
    )
    state_summary["underserved_pct"] = (state_summary["underserved"] / state_summary["total_communities"] * 100).round(1)

    fig = px.bar(state_summary.sort_values("underserved_pct", ascending=False),
                 x="state", y="underserved_pct",
                 color="underserved_pct", color_continuous_scale="Reds",
                 title="Percentage of Underserved Communities by State",
                 labels={"underserved_pct": "Underserved (%)"}, height=420)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(state_summary, x="mean_travel", y="mean_doctors",
                      size="total_population", hover_name="state",
                      color="underserved_pct", color_continuous_scale="RdYlGn_r",
                      title="Travel Time vs Doctor Density by State",
                      labels={"mean_travel": "Mean Travel Time (min)", "mean_doctors": "Doctors per 1,000"}, height=420)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(state_summary.sort_values("underserved_pct", ascending=False).reset_index(drop=True), use_container_width=True)
