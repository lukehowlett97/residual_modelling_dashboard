# layout.py

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from app import app
from styles import CUSTOM_CSS
from dash_settings import (
    residual_types, data_formats, event_labels,
    available_folders
)

# Define a color scheme
BACKGROUND_COLOR = '#1c1f26'
QUADRANT_BACKGROUND = '#23262e'
CARD_BACKGROUND = '#2d2f36'
TEXT_COLOR = '#ffffff'
ACCENT_COLOR = '#ff6600'
HEADER_BACKGROUND = '#2d2f36'

DROPDOWN_STYLE = {
    'color': '#ff6600',
    'backgroundColor': QUADRANT_BACKGROUND,
    'margin-bottom': '20px'
}

BUTTON_STYLE = {
    'backgroundColor': ACCENT_COLOR,
    'color': TEXT_COLOR,
    'border': 'none',
    'padding': '5px 10px',
    'borderRadius': '3px',
    'cursor': 'pointer'
}

CHECKLIST_LABEL_STYLE = {'color': TEXT_COLOR}

layout = dbc.Container([
    # Store Components for Sharing Data Across Callbacks
    dcc.Store(id='expanded-quadrant', data='none'),
    dcc.Store(id='files-info', data=[]),
    dcc.Store(id='selected-prns', data=[]),
    dcc.Store(id='selected-datasets', data=[]),
    dcc.Store(id='data-summary-selected-rows', data=[]),
    dcc.Store(id='data-summary-data', data=[]),

    html.Div([
        dbc.Row([
            # Top Left Quadrant
            dbc.Col([
                html.Div([
                    html.Button('Expand', id='expand-top-left', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    
                    html.H4("Select Folder", style={'color': TEXT_COLOR}),
                    dcc.Dropdown(
                        id='folder-selector',
                        options=[{'label': f, 'value': f} for f in available_folders],
                        value=available_folders[0] if available_folders else None,
                        multi=False,
                        className = 'custom-dropdown'
                    ),

                    dcc.Tabs(
                        id='top-left-tabs',
                        value='tab-file-selector',
                        children=[
                            # File Selector Tab
                            dcc.Tab(
                                label='File Selector',
                                value='tab-file-selector',
                                style={'backgroundColor': HEADER_BACKGROUND, 'color': TEXT_COLOR},
                                selected_style={'backgroundColor': HEADER_BACKGROUND, 'color': ACCENT_COLOR},
                                children=[
                                    html.H4("Data Filters", style={'margin-bottom': '20px', 'color': TEXT_COLOR}),
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col([
                                                                html.Label("Year(s):", style={'font-weight': 'bold', 'color': TEXT_COLOR}),
                                                                dcc.Dropdown(
                                                                    id='year-selector',
                                                                    options=[],
                                                                    value=[],
                                                                    multi=True,
                                                                    placeholder='Select year(s)',
                                                                    className = 'custom-dropdown'

                                                                )
                                                            ], width=4),
                                                            dbc.Col([
                                                                html.Label("DOY(s):", style={'font-weight': 'bold', 'color': TEXT_COLOR}),
                                                                dcc.Dropdown(
                                                                    id='doy-selector',
                                                                    options=[],
                                                                    value=[],
                                                                    multi=True,
                                                                    placeholder='Select DOY(s)',
                                                                    className='custom-dropdown'
                                                                )
                                                            ], width=4),
                                                            dbc.Col([
                                                                html.Label("Station(s):", style={'font-weight': 'bold', 'color': TEXT_COLOR}),
                                                                dcc.Dropdown(
                                                                    id='station-selector',
                                                                    options=[],
                                                                    value=[],
                                                                    multi=True,
                                                                    placeholder='Select station(s)',
                                                                    className='custom-dropdown'
                                                                )
                                                            ], width=4),
                                                        ],
                                                        style={'margin-bottom': '20px'}
                                                    ),
                                                    html.Label("Select PRNs:", style={'font-weight': 'bold', 'color': TEXT_COLOR}),
                                                    # prn-selection-grid should have readable text (e.g., checkboxes). 
                                                    # Add a style for any generated checkboxes/radio items:
                                                    html.Div(
                                                        id='prn-selection-grid',
                                                        style={'margin-bottom': '20px', 'color': TEXT_COLOR}
                                                    ),
                                                    html.Button(
                                                        'Add Selected PRNs to Dataset',
                                                        id='add-selected-prns-button',
                                                        n_clicks=0,
                                                        style=BUTTON_STYLE
                                                    ),
                                                    html.Hr(style={'borderColor': TEXT_COLOR}),
                                                    html.H4("Selected Datasets", style={'margin-bottom': '10px', 'color': TEXT_COLOR}),
                                                    html.Button(
                                                        'Clear All Datasets',
                                                        id='clear-selected-datasets-button',
                                                        n_clicks=0,
                                                        style=BUTTON_STYLE
                                                    ),
                                                    html.Div(
                                                        id='selected-datasets-list',
                                                        style={
                                                            'maxHeight': '200px',
                                                            'overflowY': 'auto',
                                                            'backgroundColor': QUADRANT_BACKGROUND,
                                                            'borderRadius': '3px',
                                                            'color': TEXT_COLOR
                                                        }
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={
                                            'backgroundColor': CARD_BACKGROUND,
                                            'border': '1px solid #444',
                                            'borderRadius': '5px',
                                            'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                                            'margin-bottom': '20px'
                                        }
                                    ),
                                ]
                            ),

                            # Data Summary Tab
                            dcc.Tab(
                                label='Data Summary',
                                value='tab-data-summary',
                                style={'backgroundColor': HEADER_BACKGROUND, 'color': TEXT_COLOR},
                                selected_style={'backgroundColor': HEADER_BACKGROUND, 'color': ACCENT_COLOR},
                                children=[
                                    html.Div([
                                        html.H4("Data Summary", style={'color': TEXT_COLOR}),
                                        html.Label("Select Residual:", style={'color': TEXT_COLOR}),
                                        dcc.Dropdown(
                                            id='data-summary-residual-selector',
                                            options=[],
                                            value=None,
                                            className='custom-dropdown'
                                        ),
                                        # html.Label("Select Residual Type:", style={'color': TEXT_COLOR}),
                                        # dcc.Dropdown(
                                        #     id='residual-type-selector',
                                        #     options=[],
                                        #     value=None,
                                        #     className='custom-dropdown'
                                        # ),
                                        dash_table.DataTable(
                                            id='data-summary-table',
                                            columns=[],
                                            data=[],
                                            sort_action='native',
                                            filter_action='native',
                                            page_size=20,
                                            page_action='native',
                                            row_selectable='multi',
                                            selected_rows=[],
                                            style_table={
                                                'overflowY': 'auto',
                                                'overflowX': 'auto',  # Allow horizontal scrolling
                                                'maxHeight': '50vh',
                                                'border': '1px solid #444',
                                                'width': '100%',
                                                'minWidth': '100%'
                                            },
                                            style_cell={
                                                'textAlign': 'left',
                                                'backgroundColor': QUADRANT_BACKGROUND,
                                                'color': TEXT_COLOR,
                                                'whiteSpace': 'nowrap',
                                                'overflow': 'hidden',
                                                'textOverflow': 'ellipsis',
                                                'minWidth': '80px'
                                            },
                                            style_header={
                                                'backgroundColor': HEADER_BACKGROUND,
                                                'fontWeight': 'bold',
                                                'color': TEXT_COLOR,
                                                'border': '1px solid #444',
                                                'whiteSpace': 'normal'
                                            },
                                            style_filter={
                                                'backgroundColor': HEADER_BACKGROUND,
                                                'color': TEXT_COLOR,
                                                'border': '1px solid #444'
                                            },
                                            fixed_rows={'headers': True},
                                        ),
                                        html.Br(),
                                        html.Button(
                                            'Add Selected Rows to Datasets',
                                            id='add-table-selected-rows-button',
                                            n_clicks=0,
                                            style=BUTTON_STYLE
                                        ),
                                        html.Button(
                                            'Clear Table Selections',
                                            id='clear-table-selected-rows-button',
                                            n_clicks=0,
                                            style={**BUTTON_STYLE, 'margin-left': '10px'}
                                        ),
                                        html.Hr(style={'borderColor': TEXT_COLOR}),
                                        html.H4("Selected Datasets", style={'color': TEXT_COLOR}),
                                        html.Button(
                                            'Clear All Datasets',
                                            id='clear-selected-datasets-button-data-summary',
                                            n_clicks=0,
                                            style={**BUTTON_STYLE, 'margin-bottom': '10px', 'margin-top': '10px'}
                                        ),
                                        html.Div(
                                            id='selected-datasets-list-data-summary',
                                            style={'maxHeight': '200px', 'overflowY': 'auto', 'backgroundColor': QUADRANT_BACKGROUND, 'color': TEXT_COLOR}
                                        ),
                                    ])
                                ]
                            )
                        ]
                    ),
                ], id='top-left-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '55vh', 'overflowY': 'auto', 'position': 'relative', 'backgroundColor': QUADRANT_BACKGROUND})
            ], width=4),

            # Top Right Quadrant
            dbc.Col([
                html.Div([
                    html.Button('Expand', id='expand-top-right', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    dcc.Graph(id='time-series-plot', style={'height': '100%', 'width': '100%'})
                ], id='top-right-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '55vh', 'position': 'relative', 'backgroundColor': QUADRANT_BACKGROUND})
            ], width=8),
        ], className='mb-2'),

        dbc.Row([
            # Bottom Left Quadrant
            dbc.Col([
                html.Div([
                    html.Button('Expand', id='expand-bottom-left', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    
                    html.H4("Data Configuration", style={'color': TEXT_COLOR}),
                    html.Label("Select Residual Type:", style={'color': TEXT_COLOR}),
                    dcc.Dropdown(
                        id='residual-type-selector-for-plots',
                        options=[{'label': res, 'value': res} for res in residual_types],
                        value=residual_types[0],
                        multi=False,
                        className='custom-dropdown'
                    ),
                    html.Br(),
                    
                    html.Label("Select Data Format:", style={'color': TEXT_COLOR}),
                    dcc.RadioItems(
                        id='data-format-selector',
                        options=[{'label': fmt, 'value': suffix} for fmt, suffix in data_formats.items()],
                        value='',
                        inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        style={'color': TEXT_COLOR}
                    ),
                    html.Br(),
                    
                    html.Label("Select Event Labels to Display:", style={'color': TEXT_COLOR}),
                    dbc.Checklist(
                        id='event-labels-selector',
                        options=[{'label': label.capitalize(), 'value': label} for label in event_labels],
                        value=[],
                        inline=True,
                        switch=False,
                        style={'margin-bottom': '10px'},
                        label_style=CHECKLIST_LABEL_STYLE
                    ),
                ], id='bottom-left-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '35vh', 'overflowY': 'auto', 'position': 'relative', 'backgroundColor': QUADRANT_BACKGROUND})
            ], width=4),

            # Bottom Right Quadrant
            dbc.Col([
                html.Div([
                    html.Button('Expand', id='expand-bottom-right', n_clicks=0, style=CUSTOM_CSS['expand-button']),
                    
                    html.H4("Statistics", style={'color': TEXT_COLOR}),
                    html.Div(id='statistics-output', style={'overflowY': 'auto', 'maxHeight': '30vh', 'color': TEXT_COLOR})
                ], id='bottom-right-quadrant', style={**CUSTOM_CSS['quadrant'], 'height': '35vh', 'position': 'relative', 'backgroundColor': QUADRANT_BACKGROUND})
            ], width=8),
        ]),
    ], id='quadrants-container', style=CUSTOM_CSS['quadrants-container']),
], fluid=True, style={'height': '100vh', 'backgroundColor': BACKGROUND_COLOR})