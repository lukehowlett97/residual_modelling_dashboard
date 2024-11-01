import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from pathlib import Path
from collections import defaultdict
import json  # Import json module to read JSON files

# Your main data folder path
DATA_FOLDER = Path(r"C:\Users\chcuk\Work\Projects\residual_modelling\data\processed")

def parse_filename(filename):
    if filename.endswith('.pkl'):
        # Remove the .pkl extension and split the filename
        parts = filename[:-4].split('_')
        if len(parts) >= 5:
            # The last part is system and prn, e.g., 'C06'
            system_prn = parts[-1]
            system = system_prn[0]
            prn = system_prn[1:]
            return {
                'filename': filename,
                'system': system,
                'prn': prn
            }
    return None

def list_years(data_folder, selected_folder):
    folder_path = data_folder / selected_folder
    if not folder_path.exists():
        return []
    years = [d.name for d in folder_path.iterdir() if d.is_dir()]
    return sorted(years)

def list_doys(data_folder, selected_folder, selected_years):
    doys = set()
    for year in selected_years:
        year_path = data_folder / selected_folder / year
        if year_path.is_dir():
            found_doys = [d.name for d in year_path.iterdir() if d.is_dir()]
            doys.update(found_doys)
    return sorted(doys)

def list_stations(data_folder, selected_folder, selected_years, selected_doys):
    stations = set()
    for year in selected_years:
        for doy in selected_doys:
            doy_path = data_folder / selected_folder / year / doy
            if doy_path.is_dir():
                found_stations = [d.name for d in doy_path.iterdir() if d.is_dir()]
                stations.update(found_stations)
    return sorted(stations)

def list_pkl_files(data_folder, selected_folder, selected_years, selected_doys, selected_stations):
    files = []
    for year in selected_years:
        for doy in selected_doys:
            for station in selected_stations:
                station_path = data_folder / selected_folder / year / doy / station
                if station_path.is_dir():
                    pkl_files = list(station_path.glob('*.pkl'))
                    for f in pkl_files:
                        file_info = parse_filename(f.name)
                        if file_info is None:
                            continue
                        file_info.update({
                            'filepath': str(f),
                            'year': year,
                            'doy': doy,
                            'station': station,
                            'system': file_info['system'],
                            'prn': file_info['prn']
                        })
                        files.append(file_info)
    return files

def load_data(file_paths):
    data_frames = {}
    for file_path in file_paths:
        file_path = Path(file_path)
        file_name = file_path.name
        df = pd.read_pickle(file_path)
        # Ensure 'epoch' is datetime
        if 'epoch' in df.columns:
            df['epoch'] = pd.to_datetime(df['epoch'])
        data_frames[file_name] = df
    return data_frames

def get_prns_by_system(files_info):
    prns_by_system = defaultdict(set)
    for f in files_info:
        system = f['system']
        prn = f['prn']
        prns_by_system[system].add(prn)
    # Convert sets to sorted lists
    prns_by_system = {s: sorted(list(prns)) for s, prns in prns_by_system.items()}
    return prns_by_system

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Time Series Dashboard"

available_folders = [f.name for f in DATA_FOLDER.iterdir() if f.is_dir()]

# Residual types
residual_types = ['res_oc1', 'reg_trop', 'reg_iono', 'ppprtk1']

# Data formats
data_formats = {
    'raw data': '',
    'diff': '_diff',
    'rolling mean': '_diff_rolling_mean',
    'rolling std': '_diff_rolling_std',
    'sg filter': '_sg_filter'
}

# Event labels
event_labels = ['stable', 'spikes', 'steps', 'unclassified', 'shimmering']

# Color mapping for event labels
event_colors = {
    'stable': 'LightGreen',
    'spikes': 'Red',
    'steps': 'Blue',
    'unclassified': 'Gray',
    'shimmering': 'Purple'
}


# Custom CSS styles
CUSTOM_CSS = {
    'quadrant': {
        'border': '1px solid #444',  # Darker border
        'backgroundColor': '#2c2c2c',  # Dark grey background
        'color': '#ffffff',  # White text
        'borderRadius': '5px',
        'padding': '10px',
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.2)'
    },
    'expand-button': {
        'backgroundColor': '#FF6600',  # Keep the orange accent
        'color': 'white',
        'border': 'none',
        'padding': '5px 10px',
        'borderRadius': '3px',
        'cursor': 'pointer',
        'position': 'absolute',
        'zIndex': 1001,  # Ensure it's above other elements
        'top': '10px',
        'right': '10px'
    },
    'quadrants-container': {
        'position': 'relative',
        'height': '100vh',
        'backgroundColor': '#1a1a1a',  # Darker background
        'padding': '5px'
    },
    'prn-checklist-label': {
        'display': 'inline-block',
        'width': '45px',
        'margin': '2px',
        'padding': '5px',
        'textAlign': 'center',
        'border': '1px solid #ccc',
        'borderRadius': '3px',
        'backgroundColor': '#333',
        'color': '#fff',
        'cursor': 'pointer',
    }
}

# Read the Data Summary Table
DATA_SUMMARY_PATH = DATA_FOLDER / 'processed_res20240122' / '_etc' / 'dataset_statistics.pkl'

try:
    data_summary_df = pd.read_pickle(DATA_SUMMARY_PATH)
except Exception as e:
    print(f"Error loading data summary: {e}")
    data_summary_df = pd.DataFrame()  # Empty DataFrame if loading fails

app.layout = dbc.Container([
    dcc.Store(id='expanded-quadrant', data='none'),
    dcc.Store(id='files-info', data=[]),  # Store to hold files info
    dcc.Store(id='selected-prns', data=[]),  # Store to hold selected PRNs
    dcc.Store(id='selected-datasets', data=[]),  # Store to hold selected datasets
    dcc.Store(id='data-summary-selected-rows', data=[]),  # Store for selected rows in data summary table
    html.Div([
        dbc.Row([
            # Top Left: Data/File Selector and Data Summary Tabs
            dbc.Col([
                html.Div([
                    html.Button('Expand', id='expand-top-left', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    dcc.Tabs(id='top-left-tabs', value='tab-file-selector', children=[
                        dcc.Tab(label='File Selector', value='tab-file-selector', children=[
                            html.H4("Select Data Filters"),
                            # Place Folder, Year, and DOY selectors on the same row
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Folder:"),
                                    dcc.Dropdown(
                                        id='folder-selector',
                                        options=[{'label': f, 'value': f} for f in available_folders],
                                        value=available_folders[0] if available_folders else None,
                                        multi=False,
                                        style={'color': '#000000'}
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Select Year(s):"),
                                    dcc.Dropdown(
                                        id='year-selector',
                                        options=[],
                                        value=[],
                                        multi=True,
                                        style={'color': '#000000'}
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Select DOY(s):"),
                                    dcc.Dropdown(
                                        id='doy-selector',
                                        options=[],
                                        value=[],
                                        multi=True,
                                        style={'color': '#000000'}
                                    ),
                                ], width=4),
                            ]),
                            html.Br(),
                            html.Label("Select Station(s):"),
                            dcc.Dropdown(
                                id='station-selector',
                                options=[],
                                value=[],
                                multi=True,
                                style={'color': '#000000'}
                            ),
                            html.Br(),
                            html.Hr(),
                            html.Label("Select PRNs:"),
                            html.Div(id='prn-selection-grid'),
                            html.Br(),
                            html.Button('Add Selected PRNs to Dataset', id='add-selected-prns-button', n_clicks=0),
                            html.Br(),
                            html.Hr(),
                            html.H4("Selected Datasets"),
                            html.Button('Clear All Datasets', id='clear-selected-datasets-button', n_clicks=0, style={'margin-bottom': '10px'}),
                            html.Div(id='selected-datasets-list', style={'maxHeight': '200px', 'overflowY': 'auto'}),
                        ]),
                        dcc.Tab(label='Data Summary', value='tab-data-summary', children=[
                            html.Div([
                                html.H4("Data Summary Table"),
                                dash_table.DataTable(
                                    id='data-summary-table',
                                    columns=[{"name": i, "id": i} for i in data_summary_df.columns],
                                    data=data_summary_df.to_dict('records'),
                                    sort_action='native',
                                    filter_action='native',
                                    page_size=20,
                                    page_action='native',
                                    row_selectable='multi',  # Enable row selection
                                    selected_rows=[],        # Initialize with no rows selected
                                    style_table={'overflowY': 'auto', 'maxHeight': '50vh'},
                                    style_cell={
                                        'textAlign': 'left',
                                        'minWidth': '80px',
                                        'width': '80px',
                                        'maxWidth': '180px',
                                        'whiteSpace': 'normal',
                                        'backgroundColor': '#2c2c2c',
                                        'color': 'white',
                                    },
                                    style_header={
                                        'backgroundColor': '#1a1a1a',
                                        'fontWeight': 'bold',
                                        'color': 'white'
                                    },
                                    style_filter={
                                        'backgroundColor': '#1a1a1a',
                                        'color': 'white',
                                    },
                                    fixed_rows={'headers': True},
                                ),
                                html.Br(),
                                html.Button('Add Selected Rows to Datasets', id='add-table-selected-rows-button', n_clicks=0),
                                html.Button('Clear Table Selections', id='clear-table-selected-rows-button', n_clicks=0, style={'margin-left': '10px'}),
                            ])
                        ]),
                    ]),
                ], id='top-left-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '55vh', 'overflowY': 'auto', 'position': 'relative'})
            ], width=4),
            # Top Right: Time Series Plot
            dbc.Col([
                html.Div([
                    html.Button('Expand', id='expand-top-right', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    dcc.Graph(id='time-series-plot', style={'height': '100%', 'width': '100%'})
                ], id='top-right-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '55vh', 'position': 'relative'})
            ], width=8),
        ], className='mb-2'),  # Adjusted margin
        dbc.Row([
            # Bottom Left: Data Configuration
            dbc.Col([
                html.Div([
                    html.Button('Expand', id='expand-bottom-left', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    html.H4("Data Configuration"),
                    html.Label("Select Residual Type:"),
                    dcc.Dropdown(
                        id='residual-type-selector',
                        options=[{'label': res, 'value': res} for res in residual_types],
                        value=residual_types[0],
                        multi=False,
                        style={'color': '#000000'}
                    ),
                    html.Br(),
                    html.Label("Select Data Format:"),
                    dcc.RadioItems(
                        id='data-format-selector',
                        options=[{'label': fmt, 'value': suffix} for fmt, suffix in data_formats.items()],
                        value='',
                        inputStyle={"margin-right": "5px", "margin-left": "20px"}
                    ),
                    html.Br(),
                    html.Label("Display Events:"),
                    dcc.Checklist(
                        id='display-events-checkbox',
                        options=[{'label': 'Display Events', 'value': 'display_events'}],
                        value=[],
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    ),
                    html.Br(),
                    html.Label("Select Event Labels to Display:"),
                    dcc.Dropdown(
                        id='event-labels-selector',
                        options=[{'label': label.capitalize(), 'value': label} for label in event_labels],
                        value=[],  # Default to no labels selected
                        multi=True,
                        style={'color': '#000000'}
                    ),
                ], id='bottom-left-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '35vh', 'overflowY': 'auto', 'position': 'relative'})
            ], width=4),
            # Bottom Right: Statistics Panel
            dbc.Col([
                html.Div([
                    html.Button('Expand', id='expand-bottom-right', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    html.H4("Statistics"),
                    html.Div(id='statistics-output', style={'overflowY': 'auto', 'maxHeight': '30vh'}),
                ], id='bottom-right-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '35vh', 'position': 'relative'})
            ], width=8),
        ]),
    ], id='quadrants-container', style=CUSTOM_CSS['quadrants-container']),
], fluid=True, style={'height': '100vh', 'backgroundColor': '#1a1a1a'})

# Callback to update the year selector based on selected folder
@app.callback(
    Output('year-selector', 'options'),
    Input('folder-selector', 'value')
)
def update_year_selector(selected_folder):
    if not selected_folder:
        return []
    years = list_years(DATA_FOLDER, selected_folder)
    return [{'label': y, 'value': y} for y in years]

# Callback to update the DOY selector based on selected folder and years
@app.callback(
    Output('doy-selector', 'options'),
    [Input('folder-selector', 'value'),
     Input('year-selector', 'value')]
)
def update_doy_selector(selected_folder, selected_years):
    if not selected_folder or not selected_years:
        return []
    doys = list_doys(DATA_FOLDER, selected_folder, selected_years)
    return [{'label': d, 'value': d} for d in doys]

# Callback to update the station selector based on selected folder, years, and DOYs
@app.callback(
    Output('station-selector', 'options'),
    [Input('folder-selector', 'value'),
     Input('year-selector', 'value'),
     Input('doy-selector', 'value')]
)
def update_station_selector(selected_folder, selected_years, selected_doys):
    if not selected_folder or not selected_years or not selected_doys:
        return []
    stations = list_stations(DATA_FOLDER, selected_folder, selected_years, selected_doys)
    return [{'label': s, 'value': s} for s in stations]

# Callback to update the files info store based on selected filters
@app.callback(
    Output('files-info', 'data'),
    [Input('folder-selector', 'value'),
     Input('year-selector', 'value'),
     Input('doy-selector', 'value'),
     Input('station-selector', 'value')]
)
def update_files_info(selected_folder, selected_years, selected_doys, selected_stations):
    if not selected_folder or not selected_years or not selected_doys or not selected_stations:
        return []
    files = list_pkl_files(DATA_FOLDER, selected_folder, selected_years, selected_doys, selected_stations)
    return files

# Callback to update the PRN selection grid based on available files
@app.callback(
    [Output('prn-selection-grid', 'children'),
     Output('selected-prns', 'data')],
    [Input('files-info', 'data'),
     Input({'type': 'prn-checkbox', 'index': ALL}, 'value')],
    [State({'type': 'prn-checkbox', 'index': ALL}, 'id'),
     State('selected-prns', 'data')]
)
def update_prn_selection_grid(files_info, checkbox_values, checkbox_ids, selected_prns):
    ctx = dash.callback_context
    if not files_info:
        return [html.P("No PRNs available.", style={'color': '#ffffff'})], []
    prns_by_system = get_prns_by_system(files_info)
    children = []
    new_selected_prns = set(selected_prns or [])
    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id']
        if 'prn-checkbox' in prop_id:
            for value, id_ in zip(checkbox_values, checkbox_ids):
                prn_id = id_['index']
                if value:
                    new_selected_prns.add(prn_id)
                else:
                    new_selected_prns.discard(prn_id)
    else:
        # First load, select all by default
        new_selected_prns = set(f"{sys}-{prn}" for sys, prns in prns_by_system.items() for prn in prns)
    for system in sorted(prns_by_system.keys()):
        prns = prns_by_system[system]
        checkboxes = []
        for prn in prns:
            prn_id = f"{system}-{prn}"
            is_checked = prn_id in new_selected_prns
            checkbox = dcc.Checklist(
                options=[{'label': prn, 'value': prn_id}],
                value=[prn_id] if is_checked else [],
                id={'type': 'prn-checkbox', 'index': prn_id},
                labelStyle=CUSTOM_CSS['prn-checklist-label']
            )
            checkboxes.append(checkbox)
        children.append(html.Div([
            html.H5(f"System {system}", style={'color': '#ffffff'}),
            html.Div(checkboxes, style={'display': 'flex', 'flexWrap': 'wrap'})
        ]))
    return children, list(new_selected_prns)

# Callback to update data-summary-selected-rows store and clear selections
@app.callback(
    [Output('data-summary-selected-rows', 'data'),
     Output('data-summary-table', 'selected_rows')],
    [Input('data-summary-table', 'selected_rows'),
     Input('clear-table-selected-rows-button', 'n_clicks')]
)
def update_or_clear_selected_rows(selected_rows, n_clicks_clear):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'clear-table-selected-rows-button':
        return [], []
    elif triggered_id == 'data-summary-table':
        return selected_rows, selected_rows
    else:
        raise dash.exceptions.PreventUpdate

# Callback to add selected rows and PRNs to selected datasets and handle clearing datasets
@app.callback(
    Output('selected-datasets', 'data'),
    [Input('add-table-selected-rows-button', 'n_clicks'),
     Input('add-selected-prns-button', 'n_clicks'),
     Input('clear-selected-datasets-button', 'n_clicks'),
     Input('clear-table-selected-rows-button', 'n_clicks')],
    [State('selected-datasets', 'data'),
     State('data-summary-selected-rows', 'data'),
     State('files-info', 'data'),
     State('selected-prns', 'data')]
)
def update_selected_datasets(n_clicks_table_add, n_clicks_prn_add, n_clicks_clear, n_clicks_clear_table, selected_datasets, selected_rows, files_info, selected_prns):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'add-selected-prns-button':
            if not n_clicks_prn_add or not selected_prns or not files_info:
                raise dash.exceptions.PreventUpdate
            if selected_datasets is None:
                selected_datasets = []
            existing_filepaths = set(d['filepath'] for d in selected_datasets)
            selected_prn_set = set(selected_prns)
            selected_files = [f for f in files_info if f"{f['system']}-{f['prn']}" in selected_prn_set]
            for f in selected_files:
                if f['filepath'] not in existing_filepaths:
                    selected_datasets.append(f)
            return selected_datasets
        elif button_id == 'add-table-selected-rows-button':
            if not n_clicks_table_add or not selected_rows:
                raise dash.exceptions.PreventUpdate
            if selected_datasets is None:
                selected_datasets = []
            existing_filepaths = set(d['filepath'] for d in selected_datasets)
            for idx in selected_rows:
                if idx >= len(data_summary_df):
                    continue  # Prevent out-of-range errors
                row = data_summary_df.iloc[idx]
                # Extract necessary information to build the file path
                system = row['sys']
                prn = row['prn']
                station = row['stn']
                year = str(row['year'])
                doy = str(row['doy']).zfill(3)
                folder = 'processed_res20240122'  # Need to make this dynamic at some point
                # Build the file path
                file_pattern = f"*_{system}{prn}.pkl"
                station_path = DATA_FOLDER / folder / year / doy / station
                pkl_files = list(station_path.glob(file_pattern))
                if pkl_files:
                    f = pkl_files[0]  # Assuming one file per PRN per station per day
                    file_info = parse_filename(f.name)
                    if file_info is None:
                        continue
                    file_info.update({
                        'filepath': str(f),
                        'year': year,
                        'doy': doy,
                        'station': station,
                        'system': system,
                        'prn': prn
                    })
                    if file_info['filepath'] not in existing_filepaths:
                        selected_datasets.append(file_info)
            return selected_datasets
        elif button_id == 'clear-selected-datasets-button':
            if not n_clicks_clear:
                raise dash.exceptions.PreventUpdate
            return []
        elif button_id == 'clear-table-selected-rows-button':
            if not n_clicks_clear_table:
                raise dash.exceptions.PreventUpdate
            return []
    return selected_datasets

# Update the list of selected datasets
@app.callback(
    Output('selected-datasets-list', 'children'),
    Input('selected-datasets', 'data')
)
def update_selected_datasets_list(selected_datasets):
    if not selected_datasets:
        return html.P("No datasets selected.", style={'color': '#ffffff'})
    items = []
    for f in selected_datasets:
        dataset_name = f"{f['station']} | {f['year']} | {f['doy']} | {f['system']}{f['prn']}"
        items.append(html.Div(dataset_name, style={'margin-bottom': '5px'}))
    return items

# Update the graph based on selected datasets and event configurations
# Update the graph based on selected datasets and event configurations
@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('selected-datasets', 'data'),
     Input('residual-type-selector', 'value'),
     Input('data-format-selector', 'value'),
     Input('display-events-checkbox', 'value'),
     Input('event-labels-selector', 'value')]
)
def update_graph(selected_datasets, selected_residual, data_format_suffix, display_events_value, selected_event_labels):
    if not selected_datasets or not selected_residual:
        return go.Figure()
    display_events = 'display_events' in display_events_value
    if display_events and not selected_event_labels:
        # If events are to be displayed but no labels are selected, disable event highlighting
        display_events = False
    file_paths = [f['filepath'] for f in selected_datasets]
    data_frames = load_data(file_paths)
    traces = []
    shapes = []
    annotations = []
    for f in selected_datasets:
        file_path = Path(f['filepath'])
        file_name = file_path.name
        df = data_frames.get(file_name)
        if df is None:
            continue
        x_data = df['epoch']
        column_name = selected_residual + data_format_suffix
        if column_name in df.columns:
            traces.append(go.Scatter(
                x=x_data,
                y=df[column_name],
                mode='lines',
                name=f"{f['station']} {f['year']}-{f['doy']} {f['system']}{f['prn']}"
            ))
        # Now, if display_events is True, load the seg_features and sharp_events files
        if display_events:
            # Load and process seg_features file
            seg_features_file_name = file_path.name.replace('proc_res', 'seg_features')
            seg_features_file_path = file_path.parent / seg_features_file_name
            if seg_features_file_path.exists():
                seg_features_df = pd.read_pickle(seg_features_file_path)
                # Filter events based on selected labels
                filtered_events = seg_features_df[seg_features_df['label'].isin(selected_event_labels)]
                # Get start and end indices and convert to times
                for idx, event in filtered_events.iterrows():
                    start_idx = int(event['start'])
                    end_idx = int(event['end'])
                    if start_idx >= len(df) or end_idx >= len(df):
                        continue  # Skip invalid indices
                    start_time = df['epoch'].iloc[start_idx]
                    end_time = df['epoch'].iloc[end_idx]
                    label = event['label']
                    color = event_colors.get(label, 'LightSalmon')
                    # Add shape to highlight the event
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=start_time,
                        y0=0,
                        x1=end_time,
                        y1=1,
                        fillcolor=color,
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                    ))
            # Load and process sharp_events JSON file
            sharp_events_file_name = file_path.name.replace('proc_res', 'sharp_events').replace('.pkl', '.json')
            sharp_events_file_path = file_path.parent / sharp_events_file_name
            if sharp_events_file_path.exists():
                with open(sharp_events_file_path, 'r') as json_file:
                    sharp_events_data = json.load(json_file)
                for event_type, event_data in sharp_events_data.items():
                    if event_type not in selected_event_labels:
                        continue
                    color = event_colors.get(event_type, 'LightSalmon')
                    if event_type in ['spikes', 'steps', 'unclassified']:
                        # Event data is a dictionary of indices and magnitudes
                        for idx_str in event_data:
                            idx = int(idx_str)
                            if idx >= len(df):
                                continue
                            time = df['epoch'].iloc[idx]
                            value = df[column_name].iloc[idx]
                            # Add a marker at this point
                            traces.append(go.Scatter(
                                x=[time],
                                y=[value],
                                mode='markers',
                                marker=dict(color=color, size=10, symbol='x'),
                                name=f"{event_type.capitalize()} ({f['station']} {f['system']}{f['prn']})",
                                showlegend=False
                            ))
                            # Optionally, add an annotation
                            annotations.append(dict(
                                x=time,
                                y=value,
                                xref='x',
                                yref='y',
                                text=event_type.capitalize(),
                                showarrow=True,
                                arrowhead=2,
                                ax=0,
                                ay=-20,
                                font=dict(color=color),
                                bgcolor='#2c2c2c',
                                opacity=0.8
                            ))
                    elif event_type == 'shimmering':
                        # Event data is a list of [start_idx, end_idx]
                        for start_idx, end_idx in event_data:
                            start_idx = int(start_idx)
                            end_idx = int(end_idx)
                            if start_idx >= len(df) or end_idx >= len(df):
                                continue
                            start_time = df['epoch'].iloc[start_idx]
                            end_time = df['epoch'].iloc[end_idx]
                            # Add shape to highlight the shimmering period
                            shapes.append(dict(
                                type="rect",
                                xref="x",
                                yref="paper",
                                x0=start_time,
                                y0=0,
                                x1=end_time,
                                y1=1,
                                fillcolor=color,
                                opacity=0.2,
                                layer="below",
                                line_width=0,
                            ))
        # End of display_events block
    figure = go.Figure(data=traces)
    figure.update_layout(
        plot_bgcolor='#2c2c2c',
        paper_bgcolor='#2c2c2c',
        font_color='#ffffff',
        title='Time Series Data Visualization',
        xaxis_title='Epoch',
        yaxis_title='Value',
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(
            orientation='h',
            x=1,
            xanchor='right',
            y=1.15,
            yanchor='top',
            bgcolor='rgba(30,30,30,0.8)',
            bordercolor='Black',
            borderwidth=1
        ),
        autosize=True,
        shapes=shapes,
        annotations=annotations
    )
    return figure

# Update the statistics based on selected datasets
@app.callback(
    Output('statistics-output', 'children'),
    [Input('selected-datasets', 'data'),
     Input('residual-type-selector', 'value'),
     Input('data-format-selector', 'value')]
)
def update_statistics(selected_datasets, selected_residual, data_format_suffix):
    if not selected_datasets or not selected_residual:
        return html.P("No data selected.", style={'color': '#ffffff'})

    stats_list = []
    for f in selected_datasets:
        file_path = Path(f['filepath'])
        # Build the path to the JSON file, including residual type and data format
        residual_key = selected_residual + data_format_suffix
        file_stem = file_path.stem.replace('proc_res_', '')
        json_file_name = f"sat_stats_{file_stem}.json"
        json_file_path = file_path.parent / json_file_name
        if not json_file_path.exists():
            continue  # Handle missing files appropriately
        with open(json_file_path, 'r') as json_file:
            stats_data = json.load(json_file)
        # Use stats_data directly
        residual_stats = stats_data
        # Map of original keys to shorter headers
        stats_mapping = {
            'mean': 'Mean',
            'median': 'Median',
            'std_dev': 'StdDev',
            'range': 'Range',
            'iqr': 'IQR',
            'stability_percentage': 'Stab%',
            'number_of_spikes': '#Spikes',
            'number_of_steps': '#Steps',
            'number_of_unclassified_events': '#UnclassEv',
            'number_of_shimmering_periods': '#Shimmer',
            'shimmering_percentage': 'Shimmer%',
            'mean_kurtosis': 'Kurt',
            'mean_skewness': 'Skew'
        }
        # Build the stats dictionary
        stats = {'Dataset': f"{f['station']} {f['year']}-{f['doy']} {f['system']}{f['prn']}"}
        for key, header in stats_mapping.items():
            value = residual_stats.get(key, 'N/A')
            if isinstance(value, float):
                value = f"{value:.3f}"
            stats[header] = value
        stats_list.append(stats)
    if not stats_list:
        return html.P("No statistics found for the selected datasets.", style={'color': '#ffffff'})
    stats_df = pd.DataFrame(stats_list)
    # Reorder the columns for better readability
    column_order = ['Dataset'] + list(stats_mapping.values())
    stats_df = stats_df[column_order]
    stats_table = dbc.Table.from_dataframe(stats_df, striped=True, bordered=True, hover=True, dark=True)
    return stats_table

# Expand and collapse functionality
@app.callback(
    Output('expanded-quadrant', 'data'),
    [Input('expand-top-left', 'n_clicks'),
     Input('expand-top-right', 'n_clicks'),
     Input('expand-bottom-left', 'n_clicks'),
     Input('expand-bottom-right', 'n_clicks')],
    [State('expanded-quadrant', 'data')]
)
def update_expanded_quadrant(n_clicks_tl, n_clicks_tr, n_clicks_bl, n_clicks_br, current_expanded):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'none'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # Toggle expansion
        if button_id == 'expand-top-left':
            return 'none' if current_expanded == 'top-left' else 'top-left'
        elif button_id == 'expand-top-right':
            return 'none' if current_expanded == 'top-right' else 'top-right'
        elif button_id == 'expand-bottom-left':
            return 'none' if current_expanded == 'bottom-left' else 'bottom-left'
        elif button_id == 'expand-bottom-right':
            return 'none' if current_expanded == 'bottom-right' else 'bottom-right'
    return 'none'

@app.callback(
    [Output('top-left-quadrant', 'style'),
     Output('top-right-quadrant', 'style'),
     Output('bottom-left-quadrant', 'style'),
     Output('bottom-right-quadrant', 'style')],
    [Input('expanded-quadrant', 'data')]
)
def update_quadrant_visibility(expanded):
    hidden_style = {'display': 'none'}
    expanded_style = {
        **CUSTOM_CSS['quadrant'],
        'position': 'absolute',
        'top': 0,
        'left': 0,
        'width': '100%',
        'height': '100%',
        'overflowY': 'auto',
        'zIndex': 1000,
        'backgroundColor': '#2c2c2c'
    }
    normal_style_tl = {**CUSTOM_CSS['quadrant'], 'height': '55vh', 'overflowY': 'auto', 'position': 'relative'}
    normal_style_tr = {**CUSTOM_CSS['quadrant'], 'height': '55vh', 'position': 'relative'}
    normal_style_bl = {**CUSTOM_CSS['quadrant'], 'height': '35vh', 'overflowY': 'auto', 'position': 'relative'}
    normal_style_br = {**CUSTOM_CSS['quadrant'], 'height': '35vh', 'position': 'relative'}

    if expanded == 'top-left':
        return [expanded_style, hidden_style, hidden_style, hidden_style]
    elif expanded == 'top-right':
        return [hidden_style, expanded_style, hidden_style, hidden_style]
    elif expanded == 'bottom-left':
        return [hidden_style, hidden_style, expanded_style, hidden_style]
    elif expanded == 'bottom-right':
        return [hidden_style, hidden_style, hidden_style, expanded_style]
    else:
        # No quadrant is expanded; show all quadrants in normal style
        return [normal_style_tl, normal_style_tr, normal_style_bl, normal_style_br]

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, host='0.0.0.0', port=8050)
