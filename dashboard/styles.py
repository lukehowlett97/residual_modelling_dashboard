# styles.py
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
