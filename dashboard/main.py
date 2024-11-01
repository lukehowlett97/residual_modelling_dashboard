# main.py
from app import app
from layout import layout
import callbacks  # This ensures callbacks are registered

app.layout = layout

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=True, host='0.0.0.0', port=8050)
