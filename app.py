import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

# Initialize the Dash app with a dark theme stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Outlier Detection Dashboard"

# Define the layout with dark-themed style
app.layout = html.Div(
    style={'backgroundColor': '#303030', 'color': 'white', 'minHeight': '100vh', 'padding': '20px'},
    children=[
        html.H1("Outlier Detection Dashboard", style={'textAlign': 'center'}),
        html.P("Upload a CSV file and select a detection mode to identify outliers in your data.",
               style={'textAlign': 'center'}),
        # File upload component
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'borderColor': '#aaa',
                'color': 'white',
                'margin': '10px 0'
            },
            multiple=False  # single file upload
        ),
        # Controls: dropdown for mode and button to run detection
        html.Div([
            html.Label("Detection Mode:", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='mode-dropdown',
                options=[
                    {'label': 'Simple', 'value': 'simple'},
                    {'label': 'Balanced', 'value': 'balanced'},
                    {'label': 'Complex', 'value': 'complex'}
                ],
                value='simple',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'}
            ),
            html.Button('Run Detection', id='run-button', n_clicks=0,
                        style={'margin-left': '20px', 'verticalAlign': 'middle'})
        ], style={'margin-bottom': '20px'}),
        # Output: Loading spinner, summary text, and graph
        dcc.Loading(
            id="loading",
            type="circle",
            children=[
                html.Div(id='outlier-summary', style={'margin-bottom': '10px', 'fontWeight': 'bold'}),
                dcc.Graph(id='outlier-graph')
            ]
        )
    ]
)

# Register callbacks from external module
from callbacks import register_callbacks
register_callbacks(app)

# Run the server (for local development)
if __name__ == '__main__':
    app.run_server(debug=True)
