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
    'SM06': "Chapel-Crawlspace",
}


# Explicit color ranges for KPI status by location. Update these
# ranges to control the numeric boundaries for green, yellow and red
# indicators in the KPI summary table.
KPI_COLOR_RANGES = {
    "Main": {
        "Average Temperature (°F)": {
            "green": (65, 75),
            "yellow": (55, 80),
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
            "green": (None, 45),
            "yellow": (45, 75),
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

# KPI targets derived from the green ranges above
def _build_kpi_targets(color_ranges):
    targets = {}
    for loc, metrics in color_ranges.items():
        targets[loc] = {metric: vals.get("green") for metric, vals in metrics.items()}
    return targets

KPI_TARGETS = _build_kpi_targets(KPI_COLOR_RANGES)

# --- Helper functions ---
def load_and_clean_file(path):
    """Load a device export file (.csv or .xlsx) and clean column names."""
    fn = os.path.basename(path)
    match = re.match(r"((?:AS|SM)\d+)_export_.*\.(csv|xlsx)", fn, re.IGNORECASE)
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
# Derive groups dynamically from DEVICE_LABELS so new files are
# automatically shown in the sidebar.
main = [d for d, lbl in DEVICE_LABELS.items() if lbl.endswith("-Main")]
crawlspace = [d for d, lbl in DEVICE_LABELS.items() if lbl.endswith("-Crawlspace")]
# attic = [d for d, lbl in DEVICE_LABELS.items() if lbl.endswith("-Attic")]
outdoor = [d for d, lbl in DEVICE_LABELS.items() if "Outdoor" in lbl]

location_map = {d: "Main" for d in main}
location_map.update({d: "Crawlspace" for d in crawlspace})
# location_map.update({d: "Attic" for d in attic})
location_map.update({d: "Outdoor" for d in outdoor})

# --- Streamlit App Configuration ---
st.set_page_config(page_title='St. Matthias Church: 2025 Environmental Data', layout='wide')
# Display logo
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "Logo.png")
if os.path.exists(logo_path):
    st.image(Image.open(logo_path))
else:
    st.warning(f"Logo not found at {logo_path}")

st.header('All Souls Cathedral: 2025 Environmental Data')

# Sidebar settings
st.sidebar.title('Settings')

# Data folder constant
FOLDER = './data'

# Single date range selector
date_cols = st.sidebar.columns(2)
date_cols[0].write('Start date')
# Default the start date to the beginning of the current quarter
today = datetime.today()
quarter_start_month = ((today.month - 1) // 3) * 3 + 1
current_quarter_start = datetime(today.year, quarter_start_month, 1)
start_date = date_cols[0].date_input(
    "Start Date", value=current_quarter_start, label_visibility="collapsed"
)
date_cols[1].write('End date')
end_date = date_cols[1].date_input(
    "End Date", value=datetime.today(), label_visibility="collapsed"
)

# Load and prepare data
pattern_csv = os.path.join(FOLDER, '*_export_*.csv')
pattern_xlsx = os.path.join(FOLDER, '*_export_*.xlsx')
files = glob.glob(pattern_csv) + glob.glob(pattern_xlsx)

# Read device data
device_dfs = {
    load_and_clean_file(f)['Device'].iloc[0]: load_and_clean_file(f)
    for f in files
}

records = []
if device_dfs:
    master = max(device_dfs, key=lambda d: len(device_dfs[d]))
    master_times = device_dfs[master].sort_values('Timestamp')['Timestamp']
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
# group_ui(attic, 'Attic')
group_ui(outdoor, 'Outdoor Reference')

selected_devices = [d for d in devices if st.session_state.get(f'chk_{d}', False)]

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Analysis", "Data Display", "Sensor Locations"])

with tab1:
    st.subheader("KPI (Key Performance Indicators) Summary")
    st.caption("Note: KPI targets are based on a occupied structure.")
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
            st.dataframe(styled, hide_index=True, width='stretch')

        # Targets reference table
        target_rows = []
        for loc, metrics in KPI_TARGETS.items():
            for kpi, val in metrics.items():
                if isinstance(val, tuple):
                    low, high = val
                    if low is None:
                        val_str = f"<= {high}"
                    elif high is None:
                        val_str = f">= {low}"
                    else:
                        val_str = f"{low}–{high}"
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
        if df.empty:
            st.info('No data available for the selected devices.')
        else:
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
            if 'AS10' not in selected_devices or 'AS10' not in df['Device'].unique():
                st.info('Outdoor reference data must be selected and available to display Normalized Plots')
            else:
                df_out = df[df['Device']=='AS10'][['Timestamp','Temp_F','RH']].rename(columns={'Temp_F':'T_out','RH':'RH_out'})
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
            if 'AS10' not in selected_devices or 'AS10' not in df['Device'].unique():
                st.info('Outdoor reference data must be selected and available to display Normalized Plots')
            else:
                df_norm['Norm_RH'] = df_norm['RH'] - df_norm['RH_out']
                chart_norm_rh = alt.Chart(df_norm).mark_line().encode(
                    x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
                    y=alt.Y('Norm_RH:Q', title='RH Difference (%)'),
                    color='DeviceName:N'
                )
                st.altair_chart(chart_norm_rh, use_container_width=True)

            # Pearson Corr vs Outdoor Reference
            st.header('Pearson Corr vs Outdoor Reference (Temp)')
            if 'AS10' not in selected_devices or 'AS10' not in df['Device'].unique():
                st.info('Outdoor reference data must be selected and available to display Pearson Correlation')
            else:
                cvt = compute_correlations(df, field='Temp_F')['Outdoor Reference']
                st.table(cvt.reset_index().rename(columns={'index':'DeviceName','Outdoor Reference':'Corr'}))

            st.header('Pearson Corr vs Outdoor Reference (RH)')
            if 'AS10' not in selected_devices or 'AS10' not in df['Device'].unique():
                st.info('Outdoor reference data must be selected and available to display Pearson Correlation')
            else:
                cvr = compute_correlations(df, field='RH')['Outdoor Reference']
                st.table(cvr.reset_index().rename(columns={'index':'DeviceName','Outdoor Reference':'Corr'}))

            # Summary Statistics
            st.header('Summary Statistics (Temperature)')
            st.dataframe(compute_summary_stats(df, field='Temp_F'))
            st.header('Summary Statistics (Relative Humidity)')
            st.dataframe(compute_summary_stats(df, field='RH'))

with tab3:
    st.subheader("Sensor Location Images")
    image_dir = os.path.join(script_dir, "sensor_images")
    if os.path.isdir(image_dir):
        images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not images:
            st.info("No sensor location images found.")
        else:
            for img in images:
                img_path = os.path.join(image_dir, img)
                st.image(Image.open(img_path))
                subtitle = os.path.splitext(img)[0].replace("_", " ").title()
                st.caption(subtitle)
    else:
        st.info("No sensor location images found.")
