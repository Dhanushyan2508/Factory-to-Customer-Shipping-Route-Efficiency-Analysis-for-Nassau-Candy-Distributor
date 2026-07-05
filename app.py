"""
app.py
------
Nassau Candy Distributor — Factory-to-Customer Shipping Route Efficiency
Streamlit dashboard.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then deploy on https://share.streamlit.io
    (Streamlit Community Cloud), pointing at app.py.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_engine as de
import ml_engine as ml

# --------------------------------------------------------------------------
# Page config & light styling
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Nassau Candy | Shipping Route Efficiency",
    page_icon="🍬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {background: #ffffff; border-radius: 10px; padding: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "Nassau_Candy_Distributor.csv")


# --------------------------------------------------------------------------
# Cached data loading
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and cleaning shipment data...")
def get_data(path: str):
    return de.load_and_prepare(path)


df_full, cleaning_report = get_data(DATA_PATH)

# --------------------------------------------------------------------------
# Sidebar — global filters
# --------------------------------------------------------------------------

st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Shipping Route Efficiency Analysis")
st.sidebar.divider()

min_date, max_date = df_full["Order Date"].min(), df_full["Order Date"].max()
date_range = st.sidebar.date_input(
    "Order date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

regions = sorted(df_full["Region"].unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

states = sorted(df_full["State/Province"].unique())
selected_states = st.sidebar.multiselect("State / Province (optional)", states, default=[])

ship_modes = sorted(df_full["Ship Mode"].unique())
selected_modes = st.sidebar.multiselect("Ship Mode", ship_modes, default=ship_modes)

delay_threshold = st.sidebar.slider(
    "Delay threshold (days) — shipments slower than this count as 'delayed'",
    min_value=1, max_value=int(df_full["Lead Time"].quantile(0.95)) or 10,
    value=min(5, int(df_full["Lead Time"].quantile(0.95)) or 5),
)

agg_level = st.sidebar.radio("Route granularity", ["Region", "State/Province"], horizontal=True)

with st.sidebar.expander("Data cleaning report"):
    st.json(cleaning_report)

# --------------------------------------------------------------------------
# Apply filters
# --------------------------------------------------------------------------

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start, end = min_date, max_date

mask = (
    df_full["Order Date"].between(start, end)
    & df_full["Region"].isin(selected_regions)
    & df_full["Ship Mode"].isin(selected_modes)
)
if selected_states:
    mask &= df_full["State/Province"].isin(selected_states)

df = df_full[mask].copy()

if df.empty:
    st.warning("No shipments match the current filters. Widen your selection in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------
# Header + top-line KPIs
# --------------------------------------------------------------------------

st.title("Factory → Customer Shipping Route Efficiency")
st.caption("Nassau Candy Distributor — route-level operational intelligence")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Shipments", f"{len(df):,}")
k2.metric("Avg Lead Time", f"{df['Lead Time'].mean():.1f} days")
k3.metric("Delay Rate", f"{(df['Lead Time'] > delay_threshold).mean()*100:.1f}%")
k4.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
k5.metric("Gross Profit", f"${df['Gross Profit'].sum():,.0f}")

st.divider()

tab_overview, tab_map, tab_mode, tab_drill, tab_insights = st.tabs(
    ["📊 Route Overview", "🗺️ Geographic Map", "🚚 Ship Mode Comparison",
     "🔎 Route Drill-Down", "🤖 ML Insights"]
)

# --------------------------------------------------------------------------
# TAB 1 — Route Efficiency Overview
# --------------------------------------------------------------------------

with tab_overview:
    route_summary = de.build_route_summary(df, level=agg_level, delay_threshold=delay_threshold)
    route_summary = de.score_routes(route_summary)

    st.subheader("Route performance leaderboard")
    top10 = route_summary.head(10)
    bottom10 = route_summary.tail(10).sort_values("Route_Efficiency_Score")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🏆 Top 10 most efficient routes**")
        fig = px.bar(
            top10, x="Route_Efficiency_Score", y="Route", orientation="h",
            color="Route_Efficiency_Score", color_continuous_scale="Greens",
            text="Avg_Lead_Time",
        )
        fig.update_traces(texttemplate="%{text:.1f}d avg", textposition="outside")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False,
                           height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**🐢 Bottom 10 least efficient routes**")
        fig = px.bar(
            bottom10, x="Route_Efficiency_Score", y="Route", orientation="h",
            color="Route_Efficiency_Score", color_continuous_scale="Reds_r",
            text="Avg_Lead_Time",
        )
        fig.update_traces(texttemplate="%{text:.1f}d avg", textposition="outside")
        fig.update_layout(yaxis={"categoryorder": "total descending"}, coloraxis_showscale=False,
                           height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full route table")
    st.dataframe(
        route_summary[[
            "Route", "Factory", "Location", "Route_Volume", "Avg_Lead_Time",
            "Lead_Time_StdDev", "Delay_Frequency_Pct", "Route_Efficiency_Score",
            "Total_Sales", "Total_Gross_Profit",
        ]].round(2),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Volume vs. lead time by factory")
    fig = px.scatter(
        route_summary, x="Route_Volume", y="Avg_Lead_Time", color="Factory",
        size="Route_Volume", hover_data=["Route", "Delay_Frequency_Pct"],
        labels={"Route_Volume": "Shipments on route", "Avg_Lead_Time": "Avg lead time (days)"},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 2 — Geographic Shipping Map
# --------------------------------------------------------------------------

with tab_map:
    st.subheader("US shipping efficiency heatmap")
    geo = de.get_geo_summary(df, delay_threshold=delay_threshold)

    metric_choice = st.radio(
        "Color states by", ["Avg_Lead_Time", "Shipment_Volume", "Delay_Frequency_Pct"],
        horizontal=True, key="geo_metric",
    )
    scale = "Reds" if metric_choice != "Shipment_Volume" else "Blues"

    fig = px.choropleth(
        geo, locations="State Abbr", locationmode="USA-states", color=metric_choice,
        color_continuous_scale=scale, scope="usa",
        hover_name="State/Province",
        hover_data={"Shipment_Volume": True, "Avg_Lead_Time": ":.1f", "Delay_Frequency_Pct": ":.1f", "State Abbr": False},
    )
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Canada-bound shipments are excluded from the map (no US state code) but are included everywhere else.")

    bn = ml.detect_geo_bottlenecks(geo)
    st.subheader("Congestion-prone states (high volume + high avg lead time)")
    st.dataframe(
        bn[bn["Is_Bottleneck"]][["State/Province", "Shipment_Volume", "Avg_Lead_Time", "Delay_Frequency_Pct"]].round(2),
        use_container_width=True, hide_index=True,
    )

# --------------------------------------------------------------------------
# TAB 3 — Ship Mode Comparison
# --------------------------------------------------------------------------

with tab_mode:
    st.subheader("Lead time by shipping method")
    mode_summary = de.get_ship_mode_summary(df, delay_threshold=delay_threshold)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            mode_summary, x="Ship Mode", y="Avg_Lead_Time", color="Ship Mode",
            text_auto=".1f", labels={"Avg_Lead_Time": "Avg lead time (days)"},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(
            df, x="Ship Mode", y="Lead Time", color="Ship Mode",
            points=False, labels={"Lead Time": "Lead time (days)"},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost vs. time tradeoff (descriptive)")
    fig = px.scatter(
        mode_summary, x="Avg_Lead_Time", y="Avg_Cost", color="Ship Mode",
        size="Shipment_Volume", text="Ship Mode",
        labels={"Avg_Lead_Time": "Avg lead time (days)", "Avg_Cost": "Avg cost per order ($)"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(mode_summary.round(2), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# TAB 4 — Route Drill-Down
# --------------------------------------------------------------------------

with tab_drill:
    st.subheader("State-level performance insight")
    drill_state = st.selectbox("Choose a state / province", sorted(df["State/Province"].unique()))
    state_df = df[df["State/Province"] == drill_state]

    d1, d2, d3 = st.columns(3)
    d1.metric("Shipments", f"{len(state_df):,}")
    d2.metric("Avg Lead Time", f"{state_df['Lead Time'].mean():.1f} days")
    d3.metric("Delay Rate", f"{(state_df['Lead Time'] > delay_threshold).mean()*100:.1f}%")

    fig = px.histogram(state_df, x="Lead Time", nbins=20, color="Ship Mode",
                        labels={"Lead Time": "Lead time (days)"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Order-level shipment timeline")
    timeline = state_df[[
        "Order ID", "Order Date", "Ship Date", "Lead Time", "Ship Mode",
        "Factory", "Product Name", "Sales", "Units",
    ]].sort_values("Order Date")
    st.dataframe(timeline, use_container_width=True, hide_index=True, height=400)

# --------------------------------------------------------------------------
# TAB 5 — ML Insights
# --------------------------------------------------------------------------

with tab_insights:
    st.subheader("Route performance tiers (KMeans clustering)")
    st.caption("Routes are grouped by avg lead time, variability, and delay frequency into Efficient / Moderate / Bottleneck tiers.")

    route_summary_ml = de.build_route_summary(df, level=agg_level, delay_threshold=delay_threshold)
    route_summary_ml = de.score_routes(route_summary_ml)
    clustered = ml.cluster_routes(route_summary_ml)

    fig = px.scatter(
        clustered, x="Avg_Lead_Time", y="Delay_Frequency_Pct", color="Performance_Tier",
        size="Route_Volume", hover_data=["Route"],
        color_discrete_map={"Efficient": "#2ca02c", "Moderate": "#ff9f1c", "Bottleneck": "#d62728"},
        labels={"Avg_Lead_Time": "Avg lead time (days)", "Delay_Frequency_Pct": "Delay frequency (%)"},
    )
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        clustered[["Route", "Route_Volume", "Avg_Lead_Time", "Delay_Frequency_Pct",
                   "Route_Efficiency_Score", "Performance_Tier"]].round(2),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("What drives shipping lead time?")
    st.caption("RandomForest feature importance — descriptive, explains variance already present in the data.")
    with st.spinner("Fitting model..."):
        drivers = ml.lead_time_drivers(df)
    fig = px.bar(
        drivers, x="Relative Importance", y="Factor", orientation="h",
        text_auto=".1f", color="Relative Importance", color_continuous_scale="Purples",
    )
    fig.update_layout(coloraxis_showscale=False, height=320, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built for the Nassau Candy Distributor logistics optimization project · Data through the applied filters above.")
