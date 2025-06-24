import os
import re
import glob
from datetime import datetime
from PIL import Image

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# --- Device name mapping ---
DEVICE_LABELS = {
    'SM01': "Outdoor Reference",
    'SM02': "Altar-Main",
    'SM03': "Chapel-Main",
    'SM04': "Sanctuary North-Crawlspace",
    'SM05': "Sanctuary South-Crawlspace",
}


# Explicit color ranges for KPI status by location. Update these
# ranges to control the numeric boundaries for green, yellow and red
# indicators in the KPI summary table.
KPI_COLOR_RANGES = {
    "Main": {
        "Average Temperature (°F)": {
            "green": (68, 75),
            "yellow": (60, 80),
        },
        "Temperature Swing (°F)": {
            "green": (None, 10),
            "yellow": (10, 15),
        },
        "Average Relative Humidity (%)": {
            "green": (40, 60),
            "yellow": (30, 70),
        },
        "Relative Humidity Variability (%)": {
            "green": (None, 10),
            "yellow": (10, 20),
        },
        "Average Dewpoint (°F)": {
            "green": (30, 50),
            "yellow": (20, 60),
        },
    },
    "Crawlspace": {
        "Average Temperature (°F)": {
            "green": (50, 70),
            "yellow": (40, 80),
        },
        "Temperature Swing (°F)": {
            "green": (None, 30),
            "yellow": (30, 45),
        },
        "Average Relative Humidity (%)": {
            "green": (40, 60),
            "yellow": (30, 75),
        },
        "Relative Humidity Variability (%)": {
            "green": (None, 15),
            "yellow": (15, 25),
        },
        "Average Dewpoint (°F)": {
            "green": (30, 60),
            "yellow": (25, 65),
        },
    },
    "Attic": {
        "Average Temperature (°F)": {
            "green": (50, 90),
            "yellow": (40, 100),
        },
        "Temperature Swing (°F)": {
            "green": (None, 30),
            "yellow": (30, 45),
        },
        "Average Relative Humidity (%)": {
            "green": (30, 50),
            "yellow": (30, 75),
        },
        "Relative Humidity Variability (%)": {
            "green": (None, 15),
            "yellow": (15, 25),
        },
        "Average Dewpoint (°F)": {
            "green": (30, 60),
            "yellow": (25, 65),
        },
    },
}

# KPI targets by location
KPI_TARGETS = {
    "Main": {
        "Average Temp (°F)": (68, 75),
        "Temp Swing (°F)": 5,
        "Average RH (%)": (30, 60),
        "RH Variability (%)": 10,
        "Average Dewpoint (°F)": (45, 60),
    },
    "Crawlspace": {
        "Average Temp (°F)": (60, 70),
        "Temp Swing (°F)": 7,
        "Average RH (%)": (30, 65),
        "RH Variability (%)": 15,
        "Average Dewpoint (°F)": (40, 55),
    },
    "Attic": {
        "Average Temp (°F)": (60, 80),
        "Temp Swing (°F)": 10,
        "Average RH (%)": (20, 60),
        "RH Variability (%)": 15,
        "Average Dewpoint (°F)": (30, 60),
    },
}

# --- Helper functions ---
def load_and_clean_file(path):
    """Load a device export file (.csv or .xlsx) and clean column names."""
    fn = os.path.basename(path)
    match = re.match(r"(SM\d+)_export_.*\.(csv|xlsx)", fn, re.IGNORECASE)
    device = match.group(1) if match else "Unknown"

    if fn.lower().endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={
        df.columns[0]: 'Timestamp',
        df.columns[1]: 'Temp_F',
        df.columns[2]: 'RH'
    })
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Device'] = device
    return df


def find_contiguous_nans(mask):
    gaps, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            gaps.append((start, i-1)); start = None
    if start is not None:
        gaps.append((start, len(mask)-1))
    return gaps


def fill_and_flag(series, max_gap=10, n_neighbors=4):
    orig = series.copy()
    s = series.copy()
    mask = s.isna().values
    idxs = s.index
    for start, end in find_contiguous_nans(mask):
        if (end - start + 1) <= max_gap:
            left = max(start - n_neighbors, 0)
            right = min(end + n_neighbors, len(idxs)-1)
            segment = s.iloc[left:right+1]
            s.iloc[left:right+1] = segment.interpolate()
    interpolated = orig.isna() & s.notna()
    return s, interpolated


def compute_summary_stats(df, field='Temp_F'):
    return df.groupby('DeviceName').agg(
        count=(field, 'count'),
        avg=(field, 'mean'),
        std=(field, 'std'),
        min=(field, 'min'),
        max=(field, 'max'),
        median=(field, lambda x: x.quantile(0.5)),
        missing=(field, lambda x: x.isna().sum())
    ).reset_index()


def compute_correlations(df, field='Temp_F'):
    pivot = df.pivot(index='Timestamp', columns='DeviceName', values=field)
    return pivot.corr(method='pearson')

# Calculate dewpoint in Fahrenheit from temperature (°F) and RH (%)
def dewpoint_f(temp_f, rh):
    temp_c = (temp_f - 32) * 5.0 / 9.0
    alpha = (17.27 * temp_c) / (237.7 + temp_c) + np.log(rh / 100.0)
    dew_c = (237.7 * alpha) / (17.27 - alpha)
    return dew_c * 9.0 / 5.0 + 32

# --- Device groupings for KPI Summary ---
main = [f"SM{i:02d}" for i in range(2, 4)]
crawlspace = [f"SM{i:02d}" for i in range(4, 6)]
attic = []  # placeholder for any future attic sensors

location_map = {d: "Main" for d in main}
location_map.update({d: "Crawlspace" for d in crawlspace})
location_map.update({d: "Attic" for d in attic})

# --- Streamlit App Configuration ---
st.set_page_config(page_title='St Matthias: 2025 Environmental Data', layout='wide')
# Display logo
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "Logo.png")
if os.path.exists(logo_path):
    st.image(Image.open(logo_path))
else:
    st.warning(f"Logo not found at {logo_path}")

st.header('St Matthias: 2025 Environmental Data')

# Sidebar settings
st.sidebar.title('Settings')

# Data folder constant
FOLDER = './data'

# Single date range selector
date_cols = st.sidebar.columns(2)
date_cols[0].write('Start date')
start_date = date_cols[0].date_input(
    "Start Date", value=datetime(2025, 1, 1), label_visibility="collapsed"
)
date_cols[1].write('End date')
end_date = date_cols[1].date_input(
    "End Date", value=datetime.today(), label_visibility="collapsed"
)

# Load and prepare data
pattern_csv = os.path.join(FOLDER, 'SM*_export_*.csv')
pattern_xlsx = os.path.join(FOLDER, 'SM*_export_*.xlsx')
files = glob.glob(pattern_csv) + glob.glob(pattern_xlsx)

# Read device data
device_dfs = {
    load_and_clean_file(f)['Device'].iloc[0]: load_and_clean_file(f)
    for f in files
}
# Align timestamps
master = max(device_dfs, key=lambda d: len(device_dfs[d]))
master_times = device_dfs[master].sort_values('Timestamp')['Timestamp']
records = []
for dev, df in device_dfs.items():
    tmp = df.set_index('Timestamp').reindex(
        master_times, method='nearest', tolerance=pd.Timedelta(minutes=30)
    )
    filled_t, flag_t = fill_and_flag(tmp['Temp_F'])
    filled_r, _ = fill_and_flag(tmp['RH'])
    tmp['Temp_F'] = filled_t
    tmp['RH'] = filled_r
    tmp['Dewpoint_F'] = dewpoint_f(tmp['Temp_F'], tmp['RH'])
    tmp['Interpolated'] = flag_t
    tmp['Device'] = dev
    tmp['DeviceName'] = DEVICE_LABELS.get(dev, dev)
    tmp['Location'] = location_map.get(dev, 'Outdoor')
    records.append(tmp.reset_index().rename(columns={'index': 'Timestamp'}))

# Combined DataFrame filtered by date
if records:
    df_all = pd.concat(records, ignore_index=True)
    df_all = df_all[
        (df_all['Timestamp'].dt.date >= start_date)
        & (df_all['Timestamp'].dt.date <= end_date)
    ]
else:
    df_all = pd.DataFrame()

# Device selection checkboxes for Data Display
devices = sorted(df_all['Device'].unique())
# Grouped checkboxes

def group_ui(group, label):
    st.sidebar.markdown(f'**{label}**')
    col1, col2 = st.sidebar.columns(2)
    if col1.button(f'Select All {label}'):
        for d in group:
            st.session_state[f'chk_{d}'] = True
    if col2.button(f'Deselect All {label}'):
        for d in group:
            st.session_state[f'chk_{d}'] = False
    for d in group:
        if d in devices:
            key = f'chk_{d}'
            st.session_state.setdefault(key, True)
            st.sidebar.checkbox(DEVICE_LABELS.get(d, d), key=key)

# Apply groupings
group_ui(main, 'Main')
group_ui(crawlspace, 'Crawlspace')
# Outdoor Reference
st.sidebar.markdown("**Outdoor Reference**")
for d in ['SM01']:
    if d in devices:
        key = f'chk_{d}'
        st.session_state.setdefault(key, True)
        st.sidebar.checkbox(DEVICE_LABELS.get(d, d), key=key)

selected_devices = [d for d in devices if st.session_state.get(f'chk_{d}', False)]

# Create tabs
tab1, tab2 = st.tabs(["Data Analysis", "Data Display"])

with tab1:
    st.subheader("KPI (Key Performance Indicators) Summary")
    if df_all.empty:
        st.info("No data available for the selected date range.")
    else:
        indoor_df = df_all[df_all['Location'] != 'Outdoor']

        def kpi_status(loc, kpi, val):
            ranges = KPI_COLOR_RANGES.get(loc, {}).get(kpi)
            if ranges is not None:
                g_low, g_high = ranges.get('green', (None, None))
                y_low, y_high = ranges.get('yellow', (None, None))

                def in_range(v, low, high):
                    if low is not None and v < low:
                        return False
                    if high is not None and v > high:
                        return False
                    return True

                if in_range(val, g_low, g_high):
                    return 'green'
                if in_range(val, y_low, y_high):
                    return 'yellow'
                return 'red'

            tgt = KPI_TARGETS.get(loc, {}).get(kpi)
            if tgt is None:
                return 'green'
            if isinstance(tgt, tuple):
                low, high = tgt
                if val < low or val > high:
                    if val < low - 2 or val > high + 2:
                        return 'red'
                    return 'yellow'
                return 'green'
            else:
                if val <= tgt:
                    return 'green'
                if val <= tgt + 2:
                    return 'yellow'
                return 'red'

        color_map = {'green': '#ccffcc', 'yellow': '#ffffcc', 'red': '#ffcccc'}

        for loc in ["Crawlspace", "Main", "Attic"]:
            loc_df = indoor_df[indoor_df['Location'] == loc]
            if loc_df.empty:
                continue
            avg_temp = loc_df['Temp_F'].mean()
            temp_swing = loc_df['Temp_F'].max() - loc_df['Temp_F'].min()
            avg_rh = loc_df['RH'].mean()
            rh_var = loc_df['RH'].std()
            dew = loc_df['Dewpoint_F'].mean()

            rows = []
            for kpi, val in [
                ("Average Temperature (°F)", avg_temp),
                ("Temperature Swing (°F)", temp_swing),
                ("Average Relative Humidity (%)", avg_rh),
                ("Relative Humidity Variability (%)", rh_var),
                ("Average Dewpoint (°F)", dew),
            ]:
                rows.append({"KPI": kpi, "Value": f"{val:.2f}", "Status": kpi_status(loc, kpi, val)})

            df_loc = pd.DataFrame(rows)

            def highlight(row):
                status = df_loc.loc[row.name, "Status"]
                color = color_map.get(status, "white")
                return [f"background-color: {color}; color: black"] * len(row)

            st.markdown(f"**{loc}**")
            # Hide the index and Status column in a way that works across pandas versions
            display_df = df_loc.drop(columns=["Status"])  # remove Status column
            styled = display_df.style.apply(highlight, axis=1)
            # st.dataframe supports hide_index argument from Streamlit >=1.22
            st.dataframe(styled, hide_index=True, use_container_width=True)

        # Targets reference table
        target_rows = []
        for loc, metrics in KPI_TARGETS.items():
            for kpi, val in metrics.items():
                if isinstance(val, tuple):
                    val_str = f"{val[0]}–{val[1]}"
                else:
                    val_str = f"<= {val}"
                target_rows.append({"Location": loc, "KPI": kpi, "Target": val_str})

        st.subheader("KPI Targets")
        st.table(pd.DataFrame(target_rows))

with tab2:
    st.subheader("Data Plots and Statistics")
    if df_all.empty:
        st.info("No data available for the selected date range.")
    else:
        df = df_all[df_all['Device'].isin(selected_devices)]
        # Temperature plot
        st.header('Temperature Data')
        df['DeviceName'] = df['Device'].map(DEVICE_LABELS).fillna(df['Device'])
        df_t = df.melt(id_vars=['Timestamp','DeviceName','Interpolated'], value_vars=['Temp_F'], var_name='Metric')
        line_temp = alt.Chart(df_t).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('value:Q', title='Temperature (°F)'),
            color='DeviceName:N'
        )
        pts_temp = alt.Chart(df_t[df_t['Interpolated']]).mark_circle(size=50, color='red').encode(
            x='Timestamp:T', y='value:Q'
        )
        st.altair_chart(line_temp + pts_temp, use_container_width=True)

        # Relative Humidity plot
        st.header('Relative Humidity Data')
        df_r = df.melt(id_vars=['Timestamp','DeviceName','Interpolated'], value_vars=['RH'], var_name='Metric')
        line_rh = alt.Chart(df_r).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('value:Q', title='Relative Humidity (%)'),
            color='DeviceName:N'
        )
        pts_rh = alt.Chart(df_r[df_r['Interpolated']]).mark_circle(size=50, color='red').encode(
            x='Timestamp:T', y='value:Q'
        )
        st.altair_chart(line_rh + pts_rh, use_container_width=True)

        # Correlation matrices
        st.header('Correlation Matrix (Temperature)')
        corr_t = compute_correlations(df, field='Temp_F')
        df_ct = corr_t.reset_index().rename(columns={'index':'DeviceName'}).melt(
            id_vars='DeviceName', var_name='DeviceName2', value_name='Corr'
        )
        heat_t = alt.Chart(df_ct).mark_rect().encode(
            x='DeviceName2:O', y='DeviceName:O', color='Corr:Q'
        ).properties(width=400, height=400)
        st.altair_chart(heat_t, use_container_width=False)

        st.header('Correlation Matrix (Relative Humidity)')
        corr_h = compute_correlations(df, field='RH')
        df_ch = corr_h.reset_index().rename(columns={'index':'DeviceName'}).melt(
            id_vars='DeviceName', var_name='DeviceName2', value_name='Corr'
        )
        heat_h = alt.Chart(df_ch).mark_rect().encode(
            x='DeviceName2:O', y='DeviceName:O', color='Corr:Q'
        ).properties(width=400, height=400)
        st.altair_chart(heat_h, use_container_width=False)

        # Normalized Differences
        st.header('Normalized Temperature Difference')
        df_out = df[df['Device']=='SM01'][['Timestamp','Temp_F','RH']].rename(columns={'Temp_F':'T_out','RH':'RH_out'})
        df_norm = df.merge(df_out, on='Timestamp')
        df_norm['DeviceName'] = df_norm['Device'].map(DEVICE_LABELS).fillna(df_norm['Device'])
        df_norm['Norm_T'] = df_norm['Temp_F'] - df_norm['T_out']
        chart_norm_t = alt.Chart(df_norm).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('Norm_T:Q', title='Temp Difference (°F)'),
            color='DeviceName:N'
        )
        st.altair_chart(chart_norm_t, use_container_width=True)

        st.header('Normalized Relative Humidity Difference')
        df_norm['Norm_RH'] = df_norm['RH'] - df_norm['RH_out']
        chart_norm_rh = alt.Chart(df_norm).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('Norm_RH:Q', title='RH Difference (%)'),
            color='DeviceName:N'
        )
        st.altair_chart(chart_norm_rh, use_container_width=True)

        # Pearson Corr vs Outdoor Reference
        st.header('Pearson Corr vs Outdoor Reference (Temp)')
        cvt = compute_correlations(df, field='Temp_F')['Outdoor Reference']
        st.table(cvt.reset_index().rename(columns={'index':'DeviceName','Outdoor Reference':'Corr'}))

        st.header('Pearson Corr vs Outdoor Reference (RH)')
        cvr = compute_correlations(df, field='RH')['Outdoor Reference']
        st.table(cvr.reset_index().rename(columns={'index':'DeviceName','Outdoor Reference':'Corr'}))

        # Summary Statistics
        st.header('Summary Statistics (Temperature)')
        st.dataframe(compute_summary_stats(df, field='Temp_F'))
        st.header('Summary Statistics (Relative Humidity)')
        st.dataframe(compute_summary_stats(df, field='RH'))
