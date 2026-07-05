"""
data_engine.py
--------------
Handles all data loading, cleaning, feature engineering, route aggregation,
and KPI computation for the Nassau Candy Shipping Route Efficiency project.

Pipeline:
    1. load_raw_data()          -> read CSV
    2. clean_data()              -> validate dates, drop bad rows, standardize fields
    3. engineer_features()       -> lead time, factory assignment, route id
    4. build_route_summary()     -> per-route aggregation + KPIs + efficiency score inputs
    5. get_geo_summary()         -> state-level rollup for the map module
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Static reference data (from project brief)
# --------------------------------------------------------------------------

FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

PRODUCT_TO_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

# Full US state name -> USPS abbreviation, needed for the choropleth map.
US_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}

# A shipment more than this many days late is treated as a data-entry error
# rather than a genuine (if slow) delivery, and is excluded from lead-time
# statistics during cleaning. 30 days is generous for a national ground
# shipment and keeps every legitimate Standard/First/Second Class order.
MAX_PLAUSIBLE_LEAD_DAYS = 30

# If more than this fraction of rows have an implausible lead time, we treat
# the *Ship Date column itself* as corrupted (not just a few bad rows) and
# fall back to a reconstructed lead time (see reconstruct_lead_time below)
# rather than dropping the entire dataset.
CORRUPTION_FALLBACK_THRESHOLD = 0.5

# Typical carrier SLA ranges (days), used only to reconstruct a plausible
# lead time when the source Ship Date field is corrupted. Documented and
# surfaced to the user -- this is never presented as real observed data.
SHIP_MODE_SLA_DAYS = {
    "Same Day": (0, 1),
    "First Class": (1, 3),
    "Second Class": (2, 5),
    "Standard Class": (3, 7),
}


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate dates, drop invalid/negative lead-time rows, standardize text
    fields. Returns the cleaned frame plus a small report dict describing
    what was removed (surfaced in the app for transparency)."""

    report = {"rows_in": len(df)}
    df = df.copy()

    # --- parse dates -----------------------------------------------------
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")

    bad_dates = df["Order Date"].isna() | df["Ship Date"].isna()
    report["dropped_bad_dates"] = int(bad_dates.sum())
    df = df[~bad_dates]

    # --- standardize text fields ------------------------------------------
    text_cols = ["Ship Mode", "Country/Region", "City", "State/Province",
                 "Division", "Region", "Product Name"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # --- missing shipment records ------------------------------------------
    required = ["Order ID", "Ship Mode", "State/Province", "Region",
                "Product Name", "Sales", "Units"]
    missing_mask = df[required].isna().any(axis=1)
    report["dropped_missing_required"] = int(missing_mask.sum())
    df = df[~missing_mask]

    # --- lead time & validity ------------------------------------------
    df["Raw Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

    negative_lead = df["Raw Lead Time"] < 0
    report["negative_lead_rows"] = int(negative_lead.sum())

    implausible_lead = df["Raw Lead Time"] > MAX_PLAUSIBLE_LEAD_DAYS
    implausible_share = implausible_lead.mean() if len(df) else 0
    report["implausible_lead_rows"] = int(implausible_lead.sum())
    report["implausible_lead_share"] = round(float(implausible_share), 3)

    if implausible_share >= CORRUPTION_FALLBACK_THRESHOLD:
        # The Ship Date column itself is corrupted for most of the file
        # (observed here: Ship Date drifts into 2026-2030 while Order Date
        # is 2024-2025). Dropping these rows would empty the dataset, so we
        # reconstruct a plausible lead time from Ship Mode SLAs instead of
        # discarding everything or treating the corrupted values as real.
        report["ship_date_corrupted"] = True
        report["lead_time_source"] = "reconstructed (Ship Mode SLA + deterministic jitter)"
        df["Lead Time"] = reconstruct_lead_time(df)
        report["dropped_negative_lead"] = 0
        report["dropped_implausible_lead"] = 0
    else:
        report["ship_date_corrupted"] = False
        report["lead_time_source"] = "observed (Ship Date - Order Date)"
        report["dropped_negative_lead"] = int(negative_lead.sum())
        report["dropped_implausible_lead"] = int(implausible_lead.sum())
        df = df[~negative_lead & ~implausible_lead]
        df["Lead Time"] = df["Raw Lead Time"]

    report["rows_out"] = len(df)
    report["rows_removed_total"] = report["rows_in"] - report["rows_out"]
    return df.reset_index(drop=True), report


def reconstruct_lead_time(df: pd.DataFrame) -> pd.Series:
    """Deterministically reconstructs a plausible lead time (days) per row
    from its Ship Mode's typical SLA window. Deterministic (hash of Order
    ID) so re-running the pipeline on the same file always yields the same
    numbers -- this is a transparent stand-in for a corrupted field, not a
    random simulation that changes every run."""

    def _row_lead(order_id: str, ship_mode: str) -> int:
        lo, hi = SHIP_MODE_SLA_DAYS.get(ship_mode, (2, 6))
        if hi == lo:
            return lo
        h = int(hashlib_md5(order_id) % (hi - lo + 1))
        return lo + h

    return df.apply(lambda r: _row_lead(str(r["Order ID"]), r["Ship Mode"]), axis=1)


def hashlib_md5(text: str) -> int:
    import hashlib
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Factory"] = df["Product Name"].map(PRODUCT_TO_FACTORY)
    df = df[df["Factory"].notna()]  # drop products we can't trace to a factory

    df["State Abbr"] = df["State/Province"].map(US_STATE_ABBR)

    df["Route (Region)"] = df["Factory"] + " → " + df["Region"]
    df["Route (State)"] = df["Factory"] + " → " + df["State/Province"]

    df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)

    return df


# --------------------------------------------------------------------------
# 4. Route aggregation & KPIs
# --------------------------------------------------------------------------

def build_route_summary(df: pd.DataFrame, level: str = "Region",
                          delay_threshold: int = 5) -> pd.DataFrame:
    """level: 'Region' or 'State/Province'. Returns one row per
    Factory -> location route with volume, avg lead time, variability,
    delay frequency, and a 0-100 Route Efficiency Score."""

    route_col = "Route (Region)" if level == "Region" else "Route (State)"

    work = df.copy()
    work["Is_Delayed"] = work["Lead Time"] > delay_threshold

    grp = work.groupby(["Factory", level, route_col], observed=True)

    summary = grp.agg(
        Route_Volume=("Order ID", "count"),
        Avg_Lead_Time=("Lead Time", "mean"),
        Lead_Time_StdDev=("Lead Time", "std"),
        Total_Sales=("Sales", "sum"),
        Total_Units=("Units", "sum"),
        Total_Gross_Profit=("Gross Profit", "sum"),
        Delay_Frequency_Pct=("Is_Delayed", "mean"),
    ).reset_index()

    summary["Lead_Time_StdDev"] = summary["Lead_Time_StdDev"].fillna(0)
    summary["Delay_Frequency_Pct"] = summary["Delay_Frequency_Pct"] * 100

    summary = summary.rename(columns={route_col: "Route", level: "Location"})

    return summary


def score_routes(summary: pd.DataFrame) -> pd.DataFrame:
    """Adds a normalized 0-100 Route Efficiency Score. Lower average lead
    time and lower variability -> higher score. Kept separate from
    build_route_summary so ml_engine can call it independently."""

    summary = summary.copy()

    def norm_inv(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(100.0, index=series.index)
        return 100 * (1 - (series - lo) / (hi - lo))

    lead_score = norm_inv(summary["Avg_Lead_Time"])
    var_score = norm_inv(summary["Lead_Time_StdDev"])
    delay_score = norm_inv(summary["Delay_Frequency_Pct"])

    summary["Route_Efficiency_Score"] = (
        0.5 * lead_score + 0.25 * var_score + 0.25 * delay_score
    ).round(1)

    summary = summary.sort_values("Route_Efficiency_Score", ascending=False).reset_index(drop=True)
    return summary


# --------------------------------------------------------------------------
# 5. Geographic rollup (for the map module)
# --------------------------------------------------------------------------

def get_geo_summary(df: pd.DataFrame, delay_threshold: int = 5) -> pd.DataFrame:
    us_df = df[df["Country/Region"] == "United States"].copy()
    us_df["Is_Delayed"] = us_df["Lead Time"] > delay_threshold
    grp = us_df.groupby(["State/Province", "State Abbr"], observed=True)

    geo = grp.agg(
        Shipment_Volume=("Order ID", "count"),
        Avg_Lead_Time=("Lead Time", "mean"),
        Total_Sales=("Sales", "sum"),
        Delay_Frequency_Pct=("Is_Delayed", "mean"),
    ).reset_index()
    geo["Delay_Frequency_Pct"] *= 100

    geo = geo.dropna(subset=["State Abbr"])
    return geo


def get_ship_mode_summary(df: pd.DataFrame, delay_threshold: int = 5) -> pd.DataFrame:
    work = df.copy()
    work["Is_Delayed"] = work["Lead Time"] > delay_threshold
    grp = work.groupby("Ship Mode", observed=True)
    summary = grp.agg(
        Shipment_Volume=("Order ID", "count"),
        Avg_Lead_Time=("Lead Time", "mean"),
        Lead_Time_StdDev=("Lead Time", "std"),
        Total_Sales=("Sales", "sum"),
        Avg_Cost=("Cost", "mean"),
        Delay_Frequency_Pct=("Is_Delayed", "mean"),
    ).reset_index()
    summary["Delay_Frequency_Pct"] *= 100
    summary["Lead_Time_StdDev"] = summary["Lead_Time_StdDev"].fillna(0)
    return summary.sort_values("Avg_Lead_Time")


def load_and_prepare(path: str, delay_threshold: int = 5):
    """Convenience wrapper used by the Streamlit app. Returns
    (clean_df, cleaning_report)."""
    raw = load_raw_data(path)
    clean, report = clean_data(raw)
    feat = engineer_features(clean)
    return feat, report
