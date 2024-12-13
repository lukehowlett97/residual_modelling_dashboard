# callbacks.py

from dash.dependencies import Input, Output, State, ALL
import dash
import json
import plotly.graph_objs as go
import pandas as pd
from app import app
from data_loader import (
    list_years, list_doys, list_stations, list_pkl_files, load_data, get_prns_by_system, parse_filename
)
from dash_settings import (
    DATA_FOLDER, residual_types, data_formats, event_labels, event_colors
)
from styles import CUSTOM_CSS
from pathlib import Path
from dash import html, dcc
import dash_bootstrap_components as dbc

# -------------------------------------------------------------------------
# Centralized Stats Mapping
# -------------------------------------------------------------------------
# Define a mapping from original keys to display headers
stats_mapping = {
    'station': 'Stn',
    'system': 'Sys',
    'mean': 'Mean',
    'median': 'Median',
    'std_dev': 'Std',
    'range': 'Rng',
    'iqr': 'IQR',
    'stability_percentage': 'Stab (%)',
    'number_of_spikes': 'Spikes',
    'number_of_steps': 'Steps',
    'number_of_unclassified_events': 'jumps',
    'number_of_shimmering_periods': 'Shim',
    'shimmering_percentage': 'Shim (%)',
    'mean_kurtosis': 'Kurt',
    'mean_skewness': 'Skew',
    # Add more mappings as needed
}

#--------------------------------------------------------------------------
# Callback: Update Residual Type Selector Based on Selected Folder
#--------------------------------------------------------------------------

@app.callback(
    [Output('data-summary-residual-selector', 'options'),
     Output('data-summary-residual-selector', 'value'),
    ],
    [Input('folder-selector', 'value')]
)
def update_residual_selectors(selected_folder):
    """
    Updates the options for residual selectors based on the dataset_statistics.pkl in the selected folder.
    """
    if not selected_folder:
        return [], None, [], None

    DATA_SUMMARY_PATH = Path(DATA_FOLDER) / selected_folder / '_etc' / 'dataset_statistics.pkl'

    try:
        data_summary_df = pd.read_pickle(DATA_SUMMARY_PATH)
    except Exception as e:
        print(f"Error loading data summary for folder '{selected_folder}': {e}")
        return [], None, [], None

    columns = data_summary_df.columns
    exclude_columns = ['year', 'doy', 'station', 'prn', 'system', 'prn_number']
    residuals = set()
    for col in columns:
        if col in exclude_columns:
            continue
        if '-' in col:
            residual = col.split('-')[0]
            residuals.add(residual)
    residuals = sorted(residuals)
    options = [{'label': res, 'value': res} for res in residuals]
    default_value = residuals[0] if residuals else None

    return options, default_value#, options, default_value


# -------------------------------------------------------------------------
# Callback: Update Year Selector Based on Selected Folder
# -------------------------------------------------------------------------
@app.callback(
    Output('year-selector', 'options'),
    Input('folder-selector', 'value')
)
def update_year_selector(selected_folder):
    """
    Updates the options for the Year selector based on the selected folder.
    """
    if not selected_folder:
        return []
    print(f"Selected Folder for Year Selector: {selected_folder}")
    years = list_years(DATA_FOLDER, selected_folder)
    return [{'label': y, 'value': y} for y in years]

# -------------------------------------------------------------------------
# Callback: Update DOY Selector Based on Selected Folder and Years
# -------------------------------------------------------------------------
@app.callback(
    Output('doy-selector', 'options'),
    [Input('folder-selector', 'value'),
     Input('year-selector', 'value')]
)
def update_doy_selector(selected_folder, selected_years):
    """
    Updates the options for the DOY selector based on the selected folder and years.
    """
    if not selected_folder or not selected_years:
        return []
    doys = list_doys(DATA_FOLDER, selected_folder, selected_years)
    return [{'label': d, 'value': d} for d in doys]

# -------------------------------------------------------------------------
# Callback: Update Station Selector Based on Selected Folder, Years, and DOYs
# -------------------------------------------------------------------------
@app.callback(
    Output('station-selector', 'options'),
    [Input('folder-selector', 'value'),
     Input('year-selector', 'value'),
     Input('doy-selector', 'value')]
)
def update_station_selector(selected_folder, selected_years, selected_doys):
    """
    Updates the options for the Station selector based on the selected folder, years, and DOYs.
    """
    if not selected_folder or not selected_years or not selected_doys:
        return []
    stations = list_stations(DATA_FOLDER, selected_folder, selected_years, selected_doys)
    return [{'label': s, 'value': s} for s in stations]

# -------------------------------------------------------------------------
# Callback: Update Files Info Store Based on Selected Filters
# -------------------------------------------------------------------------
@app.callback(
    Output('files-info', 'data'),
    [Input('folder-selector', 'value'),
     Input('year-selector', 'value'),
     Input('doy-selector', 'value'),
     Input('station-selector', 'value')]
)
def update_files_info(selected_folder, selected_years, selected_doys, selected_stations):
    """
    Updates the 'files-info' store with a list of available files based on selected filters.
    """
    if not selected_folder or not selected_years or not selected_doys or not selected_stations:
        return []
    files = list_pkl_files(DATA_FOLDER, selected_folder, selected_years, selected_doys, selected_stations)
    return files

# -------------------------------------------------------------------------
# Callback: Update PRN Selection Grid Based on Available Files
# -------------------------------------------------------------------------
@app.callback(
    [Output('prn-selection-grid', 'children'),
     Output('selected-prns', 'data')],
    [Input('files-info', 'data'),
     Input({'type': 'prn-checkbox', 'index': ALL}, 'value')],
    [State({'type': 'prn-checkbox', 'index': ALL}, 'id'),
     State('selected-prns', 'data')]
)
def update_prn_selection_grid(files_info, checkbox_values, checkbox_ids, selected_prns):
    """
    Updates the PRN selection grid based on available files and user interactions.
    """
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
        # On initial load, select all PRNs by default
        new_selected_prns = set(f"{sys}-{prn}" for sys, prns in prns_by_system.items() for prn in prns)
    
    for system in sorted(prns_by_system.keys()):
        prns = prns_by_system[system]
        checkboxes = []
        for prn in prns:
            prn_id = f"{system}-{prn}"
            is_checked = prn_id in new_selected_prns
            checkbox = dbc.Checklist(
                options=[{'label': prn, 'value': prn_id}],
                value=[prn_id] if is_checked else [],
                id={'type': 'prn-checkbox', 'index': prn_id},
                inline=True,
                switch=False,  # Set to False to use default checkbox appearance
                style={'margin-right': '10px', 'margin-bottom': '5px'}
            )
            checkboxes.append(checkbox)
        children.append(html.Div([
            html.H5(f"System {system}", style={'color': '#ffffff'}),
            html.Div(checkboxes, style={'display': 'flex', 'flexWrap': 'wrap'})
        ]))
    
    return children, list(new_selected_prns)

# -------------------------------------------------------------------------
# Callback: Update or Clear Selected Rows in Data Summary Table
# -------------------------------------------------------------------------
@app.callback(
    [Output('data-summary-selected-rows', 'data'),
     Output('data-summary-table', 'selected_rows')],
    [Input('data-summary-table', 'selected_rows'),
     Input('clear-table-selected-rows-button', 'n_clicks')]
)
def update_or_clear_selected_rows(selected_rows, n_clicks_clear):
    """
    Updates the store and UI based on row selection or clearing of selections in the data summary table.
    """
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

# -------------------------------------------------------------------------
# Callback: Update Selected Datasets Based on User Actions
# -------------------------------------------------------------------------
@app.callback(
    Output('selected-datasets', 'data'),
    [Input('add-table-selected-rows-button', 'n_clicks'),
     Input('add-selected-prns-button', 'n_clicks'),
     Input('clear-selected-datasets-button', 'n_clicks'),
     Input('clear-selected-datasets-button-data-summary', 'n_clicks'),
     Input('clear-table-selected-rows-button', 'n_clicks')],
    [State('selected-datasets', 'data'),
     State('data-summary-selected-rows', 'data'),
     State('files-info', 'data'),
     State('selected-prns', 'data'),
     State('data-summary-data', 'data'),
     State('folder-selector', 'value')]  # Added folder-selector as State
)
def update_selected_datasets(n_clicks_table_add, n_clicks_prn_add, n_clicks_clear, n_clicks_clear_data_summary, n_clicks_clear_table,
                             selected_datasets, selected_rows, files_info, selected_prns, data_summary, selected_folder):
    """
    Adds or clears datasets based on user interactions (adding PRNs, adding table rows, clearing datasets).
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'add-selected-prns-button':
        if not n_clicks_prn_add or not selected_prns or not files_info:
            raise dash.exceptions.PreventUpdate
        
        # Initialize the new datasets
        new_selected_datasets = selected_datasets.copy() if selected_datasets else []

        # Use a dictionary to group files by PRN
        grouped_datasets = {
            (d['system'], d['prn']): d for d in new_selected_datasets
        }

        for f in files_info:
            prn_key = (f['system'], f['prn'])
            if prn_key in grouped_datasets:
                # If the PRN already exists, update its files
                grouped_datasets[prn_key]['files'][f['filename'].split('_')[0]] = f['filepath']
            elif f"{f['system']}-{f['prn']}" in selected_prns:
                # If the PRN is new, add it
                grouped_datasets[prn_key] = {
                    'system': f['system'],
                    'prn': f['prn'],
                    'station': f['station'],
                    'year': f['year'],
                    'doy': f['doy'],
                    'files': {f['filename'].split('_')[0]: f['filepath']}
                }

        # Flatten grouped datasets back to a list
        new_selected_datasets = list(grouped_datasets.values())

        print(f"Added PRNs to datasets: {new_selected_datasets}")
        return new_selected_datasets
    
    elif button_id == 'add-table-selected-rows-button':
        if not n_clicks_table_add or not selected_rows or not data_summary:
            raise dash.exceptions.PreventUpdate

        # Create a copy to avoid in-place modification
        new_selected_datasets = selected_datasets.copy() if selected_datasets else []
        existing_filepaths = set(d['filepath'] for d in new_selected_datasets)
        # Convert data_summary (list of dicts) to DataFrame
        data_summary_df = pd.DataFrame(data_summary)
        for idx in selected_rows:
            if idx >= len(data_summary_df):
                continue  # Prevent out-of-range errors
            row = data_summary_df.iloc[idx]
            # Extract necessary information to build the file path
            system = row.get('system') or row.get('sys')  # Adjust based on actual column name
            prn = row.get('prn')
            station = row.get('station') or row.get('stn')  # Adjust based on actual column name
            year = str(row.get('year', ''))
            doy = str(row.get('doy', '')).zfill(3)
            folder = selected_folder  # Inferred from selected_folder
            if not all([system, prn, station, year, doy, folder]):
                print(f"Missing information in row {idx}: {row}")
                continue  # Skip if any necessary information is missing
            # Build the file path
            file_pattern = f"*_{system}{prn}.pkl"
            station_path = Path(DATA_FOLDER) / folder / year / doy / station
            
            
            pkl_files = list(station_path.glob(file_pattern))

            if pkl_files:
                f = pkl_files[0]  # Assuming one file per PRN per station per day
                file_info = parse_filename(f.name)
                if file_info is None:
                    print(f"Failed to parse filename: {f.name}")
                    continue
                file_info.update({
                    'filepath': str(f),
                    'year': year,
                    'doy': doy,
                    'station': station,
                    'system': system,
                    'prn': prn,
                    'folder': folder  # Added 'folder' info
                })
                if file_info['filepath'] not in existing_filepaths:
                    new_selected_datasets.append(file_info)
                    print(f"Added dataset: {file_info}")
            else:
                # print(f"No matching file found for pattern {file_pattern} in {station_path}")
                pass
        return new_selected_datasets
    
    elif button_id in ['clear-selected-datasets-button', 'clear-selected-datasets-button-data-summary']:
        if (button_id == 'clear-selected-datasets-button' and not n_clicks_clear) or \
           (button_id == 'clear-selected-datasets-button-data-summary' and not n_clicks_clear_data_summary):
            raise dash.exceptions.PreventUpdate
        print("Cleared all selected datasets.")
        return []
    
    elif button_id == 'clear-table-selected-rows-button':
        if not n_clicks_clear_table:
            raise dash.exceptions.PreventUpdate
        # Optionally, you might want to keep selected_datasets unchanged or handle accordingly
        print("Cleared table selections.")
        return selected_datasets  # Keeping datasets unchanged
    
    return selected_datasets

# -------------------------------------------------------------------------
# Callback: Update the Selected Datasets List Display
# -------------------------------------------------------------------------
@app.callback(
    [Output('selected-datasets-list', 'children'),
     Output('selected-datasets-list-data-summary', 'children')],
    Input('selected-datasets', 'data')
)
def update_selected_datasets_list(selected_datasets):
    """
    Updates the display of selected datasets in the UI without showing file paths.
    """
    if not selected_datasets:
        return (
            html.P("No datasets selected.", style={'color': '#ffffff'}),
            html.P("No datasets selected.", style={'color': '#ffffff'})
        )

    # Simplify the display: show only dataset names
    items = []
    for dataset in selected_datasets:
        dataset_name = f"{dataset['station']} | {dataset['year']}-{dataset['doy']} | {dataset['system']}{dataset['prn']}"
        items.append(html.Div(dataset_name, style={'margin-bottom': '5px'}))

    return (
        html.Div(items, style={'maxHeight': '200px', 'overflowY': 'auto'}),
        html.Div(items, style={'maxHeight': '200px', 'overflowY': 'auto'})
    )
# -------------------------------------------------------------------------
# Callback: Update the Data Summary Table Based on Selected Folder and Residual
# -------------------------------------------------------------------------
@app.callback(
    [Output('data-summary-table', 'columns'),
     Output('data-summary-table', 'data'),
     Output('data-summary-data', 'data')],
    [Input('folder-selector', 'value'),
     Input('data-summary-residual-selector', 'value')]
)
def update_data_summary(selected_folder, selected_residual):
    """
    Updates the data summary table's columns and data based on the selected folder and residual.
    Also stores the full data summary for use in other callbacks.
    """
    if not selected_folder or not selected_residual:
        return [], [], []
    
    # Construct the dynamic path based on selected folder
    DATA_SUMMARY_PATH = Path(DATA_FOLDER) / selected_folder / '_etc' / 'dataset_statistics.pkl'
    
    try:
        data_summary_df = pd.read_pickle(DATA_SUMMARY_PATH)
    except Exception as e:
        print(f"Error loading data summary for folder '{selected_folder}': {e}")
        return [], [], []  # Return empty columns and data if loading fails
    
    # Define base columns
    base_columns = ['year', 'doy', 'station', 'system', 'prn', ]
    
    if data_summary_df['year'].unique().size == 1:
        base_columns.remove('year')

    # Define residual-specific columns
    residual_columns = [col for col in data_summary_df.columns if col.startswith(selected_residual)]

    # Combine base and residual columns
    displayed_columns = base_columns + residual_columns

    # Prepare columns for DataTable using the centralized stats_mapping
    columns = []
    for col in displayed_columns:

        
        if col.startswith(selected_residual + '-'):
            key = col.replace(selected_residual + '-', '')
            display_name = stats_mapping.get(key, key.capitalize())
        else:
            display_name = stats_mapping.get(col, col.capitalize())
        columns.append({"name": display_name, "id": col})
    
    # Prepare data for DataTable
    data = data_summary_df[displayed_columns].to_dict('records')
    
    # Store the full data summary for other callbacks
    data_summary_store = data_summary_df.to_dict('records')
    
    return columns, data, data_summary_store

# -------------------------------------------------------------------------
# Callback: Update the Graph Based on Selected Datasets and Configurations
# -------------------------------------------------------------------------
@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('selected-datasets', 'data'),
     Input('residual-type-selector-for-plots', 'value'),
     Input('data-format-selector', 'value'),
     Input('event-labels-selector', 'value')]
)
def update_graph(selected_datasets, selected_residual, data_format_suffix, selected_event_labels):
    """
    Updates the time series plot based on selected datasets and user configurations.
    """
    if not selected_datasets or not selected_residual:
        return go.Figure()
    
    # Determine if events should be displayed based on selected event labels
    display_events = bool(selected_event_labels)

    print(f"selected_datasets: {selected_datasets}")
    
    file_paths = [Path(f['files']['proc']) for f in selected_datasets]

    data_frames = load_data(file_paths)
    traces = []
    shapes = []
    annotations = []
    for f in selected_datasets:
        file_path = Path(f['files']['proc'])
        if 'proc_res' not in file_path.stem:
            continue
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
        if display_events and selected_event_labels:
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

# -------------------------------------------------------------------------
# Callback: Update the Statistics Panel Based on Selected Datasets
# -------------------------------------------------------------------------
@app.callback(
    Output('statistics-output', 'children'),
    [Input('selected-datasets', 'data'),
     Input('residual-type-selector-for-plots', 'value'),
     Input('data-format-selector', 'value')]
)
def update_statistics(selected_datasets, selected_residual, data_format_suffix):
    """
    Updates the statistics panel based on the selected datasets, residual type, and data format.
    """
    if not selected_datasets:
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
            print(f"Statistics file does not exist: {json_file_path}")
            continue  # Skip if the stats JSON file does not exist
        try:
            with open(json_file_path, 'r') as json_file:
                stats_data = json.load(json_file)
        except Exception as e:
            print(f"Error loading statistics file {json_file_path}: {e}")
            continue
        
        # Extract residual-specific stats
        residual_stats = stats_data.get(residual_key, {})
        
        # Build the stats dictionary for the current dataset
        stats = {'Dataset': f"{f['station']} | {f['year']}-{f['doy']} | {f['system']}{f['prn']}"}
        for key, header in stats_mapping.items():
            value = residual_stats.get(key, 'N/A')
            if isinstance(value, float):
                value = f"{value:.3f}"
            stats[header] = value
        stats_list.append(stats)
    
    if not stats_list:
        return html.P("No statistics found for the selected datasets.", style={'color': '#ffffff'})
    
    # Create a DataFrame from the stats list
    stats_df = pd.DataFrame(stats_list)
    
    # Reorder the columns for better readability
    column_order = ['Dataset'] + [header for key, header in stats_mapping.items() if key in stats_df.columns]
    # Ensure only existing columns are included
    column_order = [col for col in column_order if col in stats_df.columns]
    stats_df = stats_df[column_order]
    
    # Create a Bootstrap Table from the DataFrame
    stats_table = dbc.Table.from_dataframe(stats_df, striped=True, bordered=True, hover=True, dark=True)
    
    return stats_table

# -------------------------------------------------------------------------
# Callback: Expand and Collapse Quadrants
# -------------------------------------------------------------------------
@app.callback(
    Output('expanded-quadrant', 'data'),
    [Input('expand-top-left', 'n_clicks'),
     Input('expand-top-right', 'n_clicks'),
     Input('expand-bottom-left', 'n_clicks'),
     Input('expand-bottom-right', 'n_clicks')],
    [State('expanded-quadrant', 'data')]
)
def update_expanded_quadrant(n_clicks_tl, n_clicks_tr, n_clicks_bl, n_clicks_br, current_expanded):
    """
    Handles the expand and collapse functionality for each quadrant.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'none'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # Toggle expansion based on which button was clicked
        if button_id == 'expand-top-left':
            return 'none' if current_expanded == 'top-left' else 'top-left'
        elif button_id == 'expand-top-right':
            return 'none' if current_expanded == 'top-right' else 'top-right'
        elif button_id == 'expand-bottom-left':
            return 'none' if current_expanded == 'bottom-left' else 'bottom-left'
        elif button_id == 'expand-bottom-right':
            return 'none' if current_expanded == 'bottom-right' else 'bottom-right'
    return 'none'

# -------------------------------------------------------------------------
# Callback: Update Quadrant Styles Based on Expansion State
# -------------------------------------------------------------------------
@app.callback(
    [Output('top-left-quadrant', 'style'),
     Output('top-right-quadrant', 'style'),
     Output('bottom-left-quadrant', 'style'),
     Output('bottom-right-quadrant', 'style')],
    [Input('expanded-quadrant', 'data')]
)
def update_quadrant_visibility(expanded):
    """
    Updates the visibility and styling of quadrants based on which one is expanded.
    """
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
