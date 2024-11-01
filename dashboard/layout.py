# layout.py

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from app import app
from styles import CUSTOM_CSS
from constants import (
    residual_types, data_formats, event_labels,
    available_folders
)

# Define the layout of the Dash application
layout = dbc.Container([
    # Store Components for Sharing Data Across Callbacks
    dcc.Store(id='expanded-quadrant', data='none'),
    dcc.Store(id='files-info', data=[]),               # Stores information about available files
    dcc.Store(id='selected-prns', data=[]),            # Stores selected PRNs
    dcc.Store(id='selected-datasets', data=[]),        # Stores selected datasets
    dcc.Store(id='data-summary-selected-rows', data=[]),  # Stores selected rows in data summary table
    dcc.Store(id='data-summary-data', data=[]),        # Stores the full data summary

    # Main Container Div
    html.Div([
        dbc.Row([
            # Top Left Quadrant: Data/File Selector and Data Summary Tabs
            dbc.Col([
                html.Div([
                    # Expand Button
                    html.Button('Expand', id='expand-top-left', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    
                    # Select Folder Dropdown
                    html.H4("Select Folder"),
                    dcc.Dropdown(
                        id='folder-selector',
                        options=[{'label': f, 'value': f} for f in available_folders],
                        value=available_folders[0] if available_folders else None,
                        multi=False,
                        style={'color': '#000000', 'margin-bottom': '20px'}
                    ),
                    
                    # Tabs for File Selector and Data Summary
                    dcc.Tabs(id='top-left-tabs', value='tab-file-selector', children=[
                        # File Selector Tab
                        dcc.Tab(label='File Selector', value='tab-file-selector', children=[
                            html.H4("Select Data Filters"),
                            
                            # Year, DOY, Station Selectors
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Year(s):"),
                                    dcc.Dropdown(
                                        id='year-selector',
                                        options=[],  # Populated via callback
                                        value=[],
                                        multi=True,
                                        style={'color': '#000000'}
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Select DOY(s):"),
                                    dcc.Dropdown(
                                        id='doy-selector',
                                        options=[],  # Populated via callback
                                        value=[],
                                        multi=True,
                                        style={'color': '#000000'}
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Select Station(s):"),
                                    dcc.Dropdown(
                                        id='station-selector',
                                        options=[],  # Populated via callback
                                        value=[],
                                        multi=True,
                                        style={'color': '#000000'}
                                    ),
                                ], width=4),
                            ]),
                            html.Br(),
                            
                            # PRN Selection Grid
                            html.Label("Select PRNs:"),
                            html.Div(id='prn-selection-grid'),
                            html.Br(),
                            
                            # Add Selected PRNs to Dataset Button
                            html.Button('Add Selected PRNs to Dataset', id='add-selected-prns-button', n_clicks=0),
                            html.Br(),
                            html.Hr(),
                            
                            # Selected Datasets Display
                            html.H4("Selected Datasets"),
                            html.Button('Clear All Datasets', id='clear-selected-datasets-button', n_clicks=0, style={'margin-bottom': '10px'}),
                            html.Div(id='selected-datasets-list', style={'maxHeight': '200px', 'overflowY': 'auto'}),
                        ]),
                        # Data Summary Tab
                        dcc.Tab(label='Data Summary', value='tab-data-summary', children=[
                            html.Div([
                                html.H4("Data Summary"),
                                
                                # Residual Selector Dropdown
                                html.Label("Select Residual:"),
                                dcc.Dropdown(
                                    id='data-summary-residual-selector',
                                    options=[
                                        {'label': 'res_oc1', 'value': 'res_oc1'},
                                        {'label': 'reg_iono', 'value': 'reg_iono'},
                                        {'label': 'reg_trop', 'value': 'reg_trop'},
                                    ],
                                    value='res_oc1',  # Default residual
                                    multi=False,
                                    style={'color': '#000000', 'margin-bottom': '20px'}
                                ),
                                
                                # Data Summary Table
                                dash_table.DataTable(
                                    id='data-summary-table',
                                    columns=[],  # Populated via callback
                                    data=[],     # Populated via callback
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
                                
                                # Add Selected Rows to Datasets Button
                                html.Button('Add Selected Rows to Datasets', id='add-table-selected-rows-button', n_clicks=0),
                                
                                # Clear Selections Button
                                html.Button('Clear Table Selections', id='clear-table-selected-rows-button', n_clicks=0, style={'margin-left': '10px'}),
                                
                                html.Hr(),
                                
                                # Selected Datasets Display in Data Summary Tab
                                html.H4("Selected Datasets"),
                                html.Button('Clear All Datasets', id='clear-selected-datasets-button-data-summary', n_clicks=0, style={'margin-bottom': '10px', 'margin-top': '10px'}),
                                html.Div(id='selected-datasets-list-data-summary', style={'maxHeight': '200px', 'overflowY': 'auto'}),
                            ])
                        ]),
                    ]),
                ], id='top-left-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '55vh', 'overflowY': 'auto', 'position': 'relative'})
            ], width=4),
            
            # Top Right Quadrant: Time Series Plot
            dbc.Col([
                html.Div([
                    # Expand Button
                    html.Button('Expand', id='expand-top-right', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    
                    # Time Series Graph
                    dcc.Graph(id='time-series-plot', style={'height': '100%', 'width': '100%'})
                ], id='top-right-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '55vh', 'position': 'relative'})
            ], width=8),
        ], className='mb-2'),  # Adjusted margin

        dbc.Row([
            # Bottom Left Quadrant: Data Configuration
            dbc.Col([
                html.Div([
                    # Expand Button
                    html.Button('Expand', id='expand-bottom-left', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    
                    # Data Configuration Title
                    html.H4("Data Configuration"),
                    
                    # Residual Type Selector
                    html.Label("Select Residual Type:"),
                    dcc.Dropdown(
                        id='residual-type-selector',
                        options=[{'label': res, 'value': res} for res in residual_types],
                        value=residual_types[0],
                        multi=False,
                        style={'color': '#000000'}
                    ),
                    html.Br(),
                    
                    # Data Format Selector
                    html.Label("Select Data Format:"),
                    dcc.RadioItems(
                        id='data-format-selector',
                        options=[{'label': fmt, 'value': suffix} for fmt, suffix in data_formats.items()],
                        value='',
                        inputStyle={"margin-right": "5px", "margin-left": "20px"}
                    ),
                    html.Br(),
                    
                    # Select Event Labels to Display (Replaced Dropdown with Checklist)
                    html.Label("Select Event Labels to Display:"),
                    dbc.Checklist(
                        id='event-labels-selector',
                        options=[{'label': label.capitalize(), 'value': label} for label in event_labels],
                        value=[],  # Default to no labels selected
                        inline=True,
                        switch=False,  # Set to False for checkbox appearance; set to True for toggle switches
                        style={'margin-bottom': '10px'}
                    ),
                ], id='bottom-left-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '35vh', 'overflowY': 'auto', 'position': 'relative'})
            ], width=4),
            
            # Bottom Right Quadrant: Statistics Panel
            dbc.Col([
                html.Div([
                    # Expand Button
                    html.Button('Expand', id='expand-bottom-right', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    
                    # Statistics Title
                    html.H4("Statistics"),
                    
                    # Statistics Output Div
                    html.Div(id='statistics-output', style={'overflowY': 'auto', 'maxHeight': '30vh'}),
                ], id='bottom-right-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '35vh', 'position': 'relative'})
            ], width=8),
        ]),
    ], id='quadrants-container', style=CUSTOM_CSS['quadrants-container']),
], fluid=True, style={'height': '100vh', 'backgroundColor': '#1a1a1a'})
