import streamlit as st
import pandas as pd
import plotly.express as px
import json
import struct
import unicodedata
import math
import os

st.set_page_config(page_title="Attica Tourism Map", layout="wide")

# ── CONFIG ────────────────────────────────────────────────────────────────────
KEY_PATH = r"C:\Users\arisg\OneDrive\Desktop\DM2key.json"
PROJECT  = "dms2-428216"
TABLE    = "dms2-428216.elstat.arrivals_overnights_occupancy"
SHP_DIR  = r"C:\Users\arisg\Downloads\attiki_shp"

NAME_OVERRIDES = {
    "Ηλιούπολης":                "ΗΛΙΟΥΠΟΛΕΩΣ",
    "Νίκαιας - Αγίου Ι. Ρέντη": "ΝΙΚΑΙΑΣ - ΑΓΙΟΥ ΙΩΑΝΝΗ ΡΕΝΤΗ",
    "Πετρούπολης":               "ΠΕΤΡΟΥΠΟΛΕΩΣ",
}

METRICS = {
    "Arrivals":       "arrivals",
    "Overnight Stays": "overnight_stays",
    "Occupancy (%)":  "fullness",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def normalize(s):
    s = s.upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip()

def ggrs87_to_wgs84(x, y):
    a, f = 6378137.0, 1/298.257222101
    b = a * (1 - f)
    e2 = (a**2 - b**2) / a**2
    e = math.sqrt(e2)
    k0 = 0.9996
    lon0 = math.radians(24.0)
    x -= 500000
    M = y / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu)
            + (151*e1**3/96)*math.sin(6*mu))
    N1 = a / math.sqrt(1 - e2*math.sin(phi1)**2)
    T1 = math.tan(phi1)**2
    C1 = e2/(1-e2) * math.cos(phi1)**2
    R1 = a*(1-e2) / (1 - e2*math.sin(phi1)**2)**1.5
    D  = x / (N1*k0)
    lat = phi1 - (N1*math.tan(phi1)/R1)*(
        D**2/2
        - (5+3*T1+10*C1-4*C1**2-9*e2/(1-e2))*D**4/24
        + (61+90*T1+298*C1+45*T1**2-252*e2/(1-e2)-3*C1**2)*D**6/720)
    lon = lon0 + (D - (1+2*T1+C1)*D**3/6
                  + (5-2*C1+28*T1-3*C1**2+8*e2/(1-e2)+24*T1**2)*D**5/120) / math.cos(phi1)
    return math.degrees(lon) + 0.0018, math.degrees(lat) - 0.0005

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
        name   = content[o:o+11].replace(b"\x00", b"").decode("latin-1").strip()
        ftype  = chr(content[o+11])
        length = content[o+16]
        fields.append((name, ftype, length))
    rows = []
    for r in range(num_records):
        ro = header_size + r*record_size + 1
        fo, row = 0, {}
        for name, ftype, length in fields:
            val = content[ro+fo:ro+fo+length]
            for enc in ["utf-8", "windows-1253", "iso-8859-7"]:
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
        if shape_type in (5, 15, 25):
            num_parts  = struct.unpack("<I", content[offset+44:offset+48])[0]
            num_points = struct.unpack("<I", content[offset+48:offset+52])[0]
            parts = [struct.unpack("<I", content[offset+52+i*4:offset+56+i*4])[0] for i in range(num_parts)]
            pts_offset = offset + 52 + num_parts*4
            points = [struct.unpack("<dd", content[pts_offset+i*16:pts_offset+i*16+16]) for i in range(num_points)]
            rings = [points[parts[p]:parts[p+1] if p+1 < num_parts else num_points] for p in range(num_parts)]
            shapes.append(rings)
        else:
            shapes.append(None)
        offset += 8 + content_len*2
    return shapes

@st.cache_data
def build_geojson(shp_dir, all_data_names):
    base = None
    for f in os.listdir(shp_dir):
        if f.endswith(".shp"):
            base = os.path.join(shp_dir, f[:-4]); break
    if not base:
        return None

    dbf_rows = read_dbf(base + ".dbf")
    shapes   = read_shp_polygons(base + ".shp")
    features = []

    for row, rings in zip(dbf_rows, shapes):
        if rings is None:
            continue
        shp_name = row.get("name", "")
        if shp_name in NAME_OVERRIDES:
            data_name = NAME_OVERRIDES[shp_name]
        else:
            norm = normalize(shp_name)
            data_name = next((dn for dn in all_data_names if normalize(dn) == norm or normalize(dn) in norm or norm in normalize(dn)), shp_name.upper())

        converted_rings = [[list(ggrs87_to_wgs84(x, y)) for x, y in ring] for ring in rings]
        geometry = {"type": "Polygon", "coordinates": [converted_rings[0]]} if len(converted_rings) == 1 \
                   else {"type": "MultiPolygon", "coordinates": [[r] for r in converted_rings]}

        features.append({
            "type": "Feature",
            "id": shp_name,
            "properties": {"shp_name": shp_name, "data_name": data_name},
            "geometry": geometry,
        })
    return {"type": "FeatureCollection", "features": features}

@st.cache_data
def load_data():
    from google.cloud import bigquery
    client = bigquery.Client.from_service_account_json(KEY_PATH)
    return client.query(f"SELECT * FROM `{TABLE}`").to_dataframe()

# ── APP ───────────────────────────────────────────────────────────────────────
st.title("🗺️ Attica Tourism Dashboard")

with st.spinner("Loading data from BigQuery..."):
    df = load_data()

# Filter to Attica
df = df[df["region"].str.contains("ΑΤΤΙΚ", na=False)].copy()
df["municipality"] = df["municipality"].str.strip().str.replace(r"^ΔΗΜΟ[ΣΙ]\s+", "", regex=True)

all_data_names = sorted(df["municipality"].dropna().unique().tolist())

# Sidebar
st.sidebar.header("Filters")
selected_metric_label = st.sidebar.selectbox("Metric", list(METRICS.keys()))
selected_metric_col   = METRICS[selected_metric_label]
years    = sorted(df["year"].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox("Year", years)
sources  = ["All"] + sorted(df["source"].dropna().unique())
selected_source = st.sidebar.selectbox("Source", sources)

# Filter
fdf = df[df["year"] == selected_year].copy()
if selected_source != "All":
    fdf = fdf[fdf["source"] == selected_source]

# Aggregate
agg = fdf.groupby("municipality")[selected_metric_col].sum().reset_index()
agg.columns = ["data_name", "value"]

# Build GeoJSON
with st.spinner("Building map..."):
    geojson = build_geojson(SHP_DIR, tuple(all_data_names))

if not geojson:
    st.error(f"Shapefile not found in: {SHP_DIR}")
    st.stop()

# Attach values
name_to_val = dict(zip(agg["data_name"], agg["value"]))
map_df = pd.DataFrame([{
    "shp_name":  f["properties"]["shp_name"],
    "data_name": f["properties"]["data_name"],
    "value":     name_to_val.get(f["properties"]["data_name"], 0),
} for f in geojson["features"]])

# Map
fig = px.choropleth_mapbox(
    map_df,
    geojson=geojson,
    locations="shp_name",
    featureidkey="id",
    color="value",
    color_continuous_scale="Blues",
    mapbox_style="carto-positron",
    zoom=8.5,
    center={"lat": 37.97, "lon": 23.73},
    opacity=0.75,
    hover_name="shp_name",
    hover_data={"value": ":,.0f", "shp_name": False},
    labels={"value": selected_metric_label},
    title=f"{selected_metric_label} by Municipality — {selected_year}",
)
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=650)
st.plotly_chart(fig, use_container_width=True)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Arrivals",       f"{fdf['arrivals'].sum():,.0f}")
col2.metric("Total Overnight Stays", f"{fdf['overnight_stays'].sum():,.0f}")
col3.metric("Avg Occupancy",         f"{fdf['fullness'].mean():.1f}%")

# Table
with st.expander("Data table"):
    st.dataframe(agg.sort_values("value", ascending=False), use_container_width=True)
