import streamlit as st
import pandas as pd
from google.cloud import bigquery

st.set_page_config(page_title="Regional Dashboard", layout="wide")

KEY_PATH = r"C:\Users\arisg\OneDrive\Desktop\DM2key.json"
PROJECT = "dms2-428216"
VIEW = "dms2-428216.analytics.regional_quarterly_enriched"

@st.cache_data
def load_data():
    client = bigquery.Client.from_service_account_json(KEY_PATH)
    query = f"SELECT * FROM `{VIEW}`"
    return client.query(query).to_dataframe()

df = load_data()

st.title("Regional Quarterly Dashboard")

# Sidebar filters
regions = sorted(df["region"].dropna().unique())
metrics = sorted(df["metric"].dropna().unique())

selected_region = st.sidebar.multiselect("Region", regions, default=regions)
selected_metric = st.sidebar.selectbox("Metric", metrics)

filtered = df[
    (df["region"].isin(selected_region)) &
    (df["metric"] == selected_metric)
].sort_values("period_date")

# KPIs - latest period
latest = filtered[filtered["is_latest"] == 1]

st.subheader(f"{selected_metric} — Latest Period")
cols = st.columns(len(latest["region"].unique()) if len(latest) > 0 else 1)
for i, (_, row) in enumerate(latest.iterrows()):
    if i < len(cols):
        cols[i].metric(
            label=row["region"],
            value=f"{row['value']:,}",
            delta=f"{row['yoy_pct']:.1f}% YoY" if pd.notna(row["yoy_pct"]) else None
        )

st.divider()

# Trend chart
st.subheader("Trend Over Time")
import plotly.express as px

chart_df = filtered.groupby(["period_date", "region"])["value"].sum().reset_index()
fig = px.line(
    chart_df,
    x="period_date",
    y="value",
    color="region",
    markers=True,
    title=f"{selected_metric} by Region"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# YoY comparison
st.subheader("Year-over-Year % Change")
yoy_df = filtered[filtered["yoy_pct"].notna()].copy()
fig2 = px.bar(
    yoy_df,
    x="period_date",
    y="yoy_pct",
    color="region",
    barmode="group",
    title="YoY % Change by Quarter"
)
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Raw data
with st.expander("View raw data"):
    st.dataframe(filtered, use_container_width=True)