import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import struct
import unicodedata
import math
import os
import numpy as np

st.set_page_config(page_title="Attica Tourism Map", layout="wide", initial_sidebar_state="expanded")

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1c1f26; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #9aa0b0 !important; font-size: 13px !important; }
    .stMetric [data-testid="metric-container"] > div { color: white !important; }
    h1, h2, h3 { color: white !important; }
    .note-box {
        background-color: #1c2333;
        border-left: 3px solid #4a90d9;
        padding: 10px 14px;
        border-radius: 4px;
        color: #9aa0b0;
        font-size: 13px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT = "dms2-428216"
TABLE   = "dms2-428216.elstat.arrivals_overnights_occupancy"
SHP_DIR = "shapefiles"

NAME_OVERRIDES = {
    "Ηλιούπολης":                "ΗΛΙΟΥΠΟΛΕΩΣ",
    "Νίκαιας - Αγίου Ι. Ρέντη": "ΝΙΚΑΙΑΣ - ΑΓΙΟΥ ΙΩΑΝΝΗ ΡΕΝΤΗ",
    "Πετρούπολης":               "ΠΕΤΡΟΥΠΟΛΕΩΣ",
}

METRICS = {
    "Arrivals":        {"col": "arrivals",        "color": "YlOrRd", "fmt": ",.0f"},
    "Overnight Stays": {"col": "overnight_stays", "color": "YlOrRd", "fmt": ",.0f"},
    "Occupancy (%)":   {"col": "fullness",        "color": "RdYlGn", "fmt": ".1f"},
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def normalize(s):
    s = s.upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip()

def ggrs87_to_wgs84(x, y):
    a, f = 6378137.0, 1/298.257222101
    e2 = 1 - (a*(1-f)/a)**2 + (2*f - f**2)
    e2 = (2*f - f**2)
    k0, lon0 = 0.9996, math.radians(24.0)
    x -= 500000
    M = y / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1 = (1 - math.sqrt(1-e2)) / (1 + math.sqrt(1-e2))
    phi1 = (mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu)
            + (151*e1**3/96)*math.sin(6*mu))
    N1 = a / math.sqrt(1 - e2*math.sin(phi1)**2)
    T1 = math.tan(phi1)**2
    C1 = e2/(1-e2) * math.cos(phi1)**2
    R1 = a*(1-e2) / (1 - e2*math.sin(phi1)**2)**1.5
    D  = x / (N1*k0)
    lat = phi1 - (N1*math.tan(phi1)/R1)*(
        D**2/2 - (5+3*T1+10*C1-4*C1**2-9*e2/(1-e2))*D**4/24
        + (61+90*T1+298*C1+45*T1**2-252*e2/(1-e2)-3*C1**2)*D**6/720)
    lon = lon0 + (D - (1+2*T1+C1)*D**3/6
                  + (5-2*C1+28*T1-3*C1**2+8*e2/(1-e2)+24*T1**2)*D**5/120) / math.cos(phi1)
    return math.degrees(lon)+0.0018, math.degrees(lat)-0.0005

def read_dbf(path):
    with open(path, "rb") as f:
        content = f.read()
    num_records = struct.unpack("<I", content[4:8])[0]
    header_size = struct.unpack("<H", content[8:10])[0]
    record_size = struct.unpack("<H", content[10:12])[0]
    num_fields  = (header_size - 32 - 1) // 32
    fields = []
    for i in range(num_fields):
        o = 32 + i*32
        name   = content[o:o+11].replace(b"\x00",b"").decode("latin-1").strip()
        length = content[o+16]
        fields.append((name, length))
    rows = []
    for r in range(num_records):
        ro, fo, row = header_size + r*record_size + 1, 0, {}
        for name, length in fields:
            val = content[ro+fo:ro+fo+length]
            for enc in ["utf-8","windows-1253","iso-8859-7"]:
                try: row[name] = val.decode(enc).strip(); break
                except: pass
            fo += length
        rows.append(row)
    return rows

def read_shp_polygons(shp_path):
    with open(shp_path, "rb") as f:
        content = f.read()
    offset, shapes = 100, []
    while offset < len(content):
        content_len = struct.unpack(">I", content[offset+4:offset+8])[0]
        shape_type  = struct.unpack("<I", content[offset+8:offset+12])[0]
        if shape_type in (5,15,25):
            num_parts  = struct.unpack("<I", content[offset+44:offset+48])[0]
            num_points = struct.unpack("<I", content[offset+48:offset+52])[0]
            parts = [struct.unpack("<I", content[offset+52+i*4:offset+56+i*4])[0] for i in range(num_parts)]
            pts_o  = offset + 52 + num_parts*4
            points = [struct.unpack("<dd", content[pts_o+i*16:pts_o+i*16+16]) for i in range(num_points)]
            rings  = [points[parts[p]:parts[p+1] if p+1<num_parts else num_points] for p in range(num_parts)]
            shapes.append(rings)
        else:
            shapes.append(None)
        offset += 8 + content_len*2
    return shapes

@st.cache_data
def build_geojson(shp_dir, all_data_names):
    base = next((os.path.join(shp_dir, f[:-4]) for f in os.listdir(shp_dir) if f.endswith(".shp")), None)
    if not base: return None
    dbf_rows = read_dbf(base+".dbf")
    shapes   = read_shp_polygons(base+".shp")
    features = []
    for row, rings in zip(dbf_rows, shapes):
        if rings is None: continue
        shp_name = row.get("name","")
        if shp_name in NAME_OVERRIDES:
            data_name = NAME_OVERRIDES[shp_name]
        else:
            norm = normalize(shp_name)
            data_name = next((dn for dn in all_data_names
                              if normalize(dn)==norm or normalize(dn) in norm or norm in normalize(dn)),
                             shp_name.upper())
        converted = [[list(ggrs87_to_wgs84(x,y)) for x,y in ring] for ring in rings]
        geometry  = ({"type":"Polygon","coordinates":[converted[0]]} if len(converted)==1
                     else {"type":"MultiPolygon","coordinates":[[r] for r in converted]})
        features.append({"type":"Feature","id":shp_name,
                         "properties":{"shp_name":shp_name,"data_name":data_name},
                         "geometry":geometry})
    return {"type":"FeatureCollection","features":features}

@st.cache_data
def load_data():
    from google.cloud import bigquery
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]))
    client = bigquery.Client(credentials=credentials, project=PROJECT)
    return client.query(f"SELECT * FROM `{TABLE}`").to_dataframe()

# ── APP ───────────────────────────────────────────────────────────────────────
st.title("🗺️ Attica Tourism Dashboard")
st.markdown("""<div class='note-box'>
📊 Data source: ELSTAT — Municipality-level groupings vary by year due to statistical reporting changes.
Some areas appear combined (e.g. Athens + Byron + Nea Filadelfia) depending on the selected year.
</div>""", unsafe_allow_html=True)

with st.spinner("Loading data from BigQuery..."):
    df = load_data()

df = df[df["region"].str.contains("ΑΤΤΙΚ", na=False)].copy()
df["municipality"] = df["municipality"].str.strip().str.replace(r"^ΔΗΜΟ[ΣΙ]\s+", "", regex=True)
all_data_names = tuple(sorted(df["municipality"].dropna().unique().tolist()))

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_metric_label = st.sidebar.selectbox("Metric", list(METRICS.keys()))
metric_cfg  = METRICS[selected_metric_label]
metric_col  = metric_cfg["col"]
years = sorted(df["year"].dropna().unique(), reverse=True)
selected_year = st.sidebar.select_slider("Year", options=sorted(df["year"].dropna().unique()), value=max(years))
sources = ["All"] + sorted(df["source"].dropna().unique())
selected_source = st.sidebar.selectbox("Source", sources)
map_style = st.sidebar.selectbox("Map Style", ["carto-darkmatter","carto-positron","open-street-map"])
use_log = st.sidebar.checkbox("Log scale (reduces Athens dominance)", value=True)

# ── FILTER ────────────────────────────────────────────────────────────────────
fdf = df[df["year"] == selected_year].copy()
if selected_source != "All":
    fdf = fdf[fdf["source"] == selected_source]

agg = fdf.groupby("municipality").agg(
    arrivals=("arrivals","sum"),
    overnight_stays=("overnight_stays","sum"),
    fullness=("fullness","mean")
).reset_index()

# ── KPIS ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Arrivals",        f"{agg['arrivals'].sum():,.0f}")
k2.metric("Total Overnight Stays", f"{agg['overnight_stays'].sum():,.0f}")
k3.metric("Avg Occupancy",         f"{agg['fullness'].mean():.1f}%")
k4.metric("Municipalities",        f"{agg['municipality'].nunique()}")

st.divider()

# ── GEOJSON ───────────────────────────────────────────────────────────────────
with st.spinner("Rendering map..."):
    geojson = build_geojson(SHP_DIR, all_data_names)

if not geojson:
    st.error(f"Shapefile not found in: {SHP_DIR}")
    st.stop()

name_to_row = {r["municipality"]: r for _, r in agg.iterrows()}
map_df = pd.DataFrame([{
    "shp_name":       f["properties"]["shp_name"],
    "data_name":      f["properties"]["data_name"],
    "arrivals":       name_to_row.get(f["properties"]["data_name"], {}).get("arrivals", None),
    "overnight_stays":name_to_row.get(f["properties"]["data_name"], {}).get("overnight_stays", None),
    "fullness":       name_to_row.get(f["properties"]["data_name"], {}).get("fullness", None),
    "has_data":       f["properties"]["data_name"] in name_to_row,
} for f in geojson["features"]])

map_df["value"] = map_df[metric_col]
map_df["value_display"] = map_df["value"]
if use_log and metric_col != "fullness":
    map_df["value_plot"] = np.log1p(map_df["value"].fillna(0))
    colorbar_title = f"{selected_metric_label} (log scale)"
else:
    map_df["value_plot"] = map_df["value"].fillna(0)
    colorbar_title = selected_metric_label

# ── MAP ───────────────────────────────────────────────────────────────────────
map_col, bar_col = st.columns([3, 1])

with map_col:
    fig_map = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="shp_name",
        featureidkey="id",
        color="value_plot",
        color_continuous_scale=metric_cfg["color"],
        mapbox_style=map_style,
        zoom=8.5,
        center={"lat": 37.97, "lon": 23.73},
        opacity=0.8,
        hover_name="shp_name",
        custom_data=["value_display", "has_data"],
        labels={"value_plot": colorbar_title},
        title=f"{selected_metric_label} — {selected_year}",
    )
    fig_map.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      f"{selected_metric_label}: " + "%{customdata[0]:,.0f}<br>" +
                      "<extra></extra>"
    )
    fig_map.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=580,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        coloraxis_colorbar=dict(title=dict(text=colorbar_title, font=dict(color="white")), tickfont=dict(color="white")),

    )
    st.plotly_chart(fig_map, use_container_width=True)

# ── TOP 10 BAR CHART ──────────────────────────────────────────────────────────
with bar_col:
    st.markdown(f"#### Top 10 — {selected_metric_label}")
    top10 = map_df[map_df["has_data"]].nlargest(10, "value")[["shp_name","value"]].reset_index(drop=True)
    top10["rank"] = top10.index + 1
    fig_bar = go.Figure(go.Bar(
        x=top10["value"],
        y=top10["shp_name"],
        orientation="h",
        marker_color="#4a90d9",
        text=top10["value"].apply(lambda v: f"{v:,.0f}"),
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig_bar.update_layout(
        height=580,
        margin={"r":20,"t":10,"l":10,"b":10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── DETAIL ON CLICK (selected municipality breakdown) ─────────────────────────
st.divider()
st.subheader("Municipality Breakdown")
selected_mun = st.selectbox("Select municipality", ["— select —"] + sorted(map_df[map_df["has_data"]]["shp_name"].tolist()))

if selected_mun != "— select —":
    row = map_df[map_df["shp_name"] == selected_mun].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Arrivals",        f"{row['arrivals']:,.0f}"  if pd.notna(row['arrivals'])  else "N/A")
    c2.metric("Overnight Stays", f"{row['overnight_stays']:,.0f}" if pd.notna(row['overnight_stays']) else "N/A")
    c3.metric("Avg Occupancy",   f"{row['fullness']:.1f}%"  if pd.notna(row['fullness'])  else "N/A")

    # Trend over all years for this municipality
    mun_data_name = map_df[map_df["shp_name"]==selected_mun]["data_name"].values[0]
    trend = df[df["municipality"]==mun_data_name].copy()
    if selected_source != "All":
        trend = trend[trend["source"]==selected_source]
    trend_agg = trend.groupby("year").agg(
        arrivals=("arrivals","sum"),
        overnight_stays=("overnight_stays","sum"),
        fullness=("fullness","mean")
    ).reset_index()

    if not trend_agg.empty:
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(go.Bar(x=trend_agg["year"], y=trend_agg["arrivals"],
                                   name="Arrivals", marker_color="#4a90d9"), secondary_y=False)
        fig_trend.add_trace(go.Bar(x=trend_agg["year"], y=trend_agg["overnight_stays"],
                                   name="Overnight Stays", marker_color="#7bc8f6"), secondary_y=False)
        fig_trend.add_trace(go.Scatter(x=trend_agg["year"], y=trend_agg["fullness"],
                                       name="Occupancy %", mode="lines+markers",
                                       line=dict(color="#f0a500", width=2)), secondary_y=True)
        fig_trend.update_layout(
            title=f"{selected_mun} — All Years",
            barmode="group",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(font=dict(color="white")),
        )
        fig_trend.update_yaxes(title_text="Count", secondary_y=False, gridcolor="#2a2d35")
        fig_trend.update_yaxes(title_text="Occupancy %", secondary_y=True, gridcolor="#2a2d35")
        st.plotly_chart(fig_trend, use_container_width=True)

# ── RAW DATA ──────────────────────────────────────────────────────────────────
with st.expander("📋 Full data table"):
    display = map_df[map_df["has_data"]][["shp_name","arrivals","overnight_stays","fullness"]].copy()
    display.columns = ["Municipality","Arrivals","Overnight Stays","Occupancy %"]
    display = display.sort_values("Arrivals", ascending=False)
    st.dataframe(display, use_container_width=True)

st.caption(f"Source: ELSTAT | Year: {selected_year} | Source filter: {selected_source}")
