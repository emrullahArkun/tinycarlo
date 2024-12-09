from dash import Dash, dcc, html
import dash
import dash_bootstrap_components as dbc

# Use a Bootstrap theme
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CYBORG])

# Layout of the application
app.layout = html.Div([
    dcc.Location(id="url"),  # Monitors the URL of the application

    # Navigation bar with links to the pages
    dbc.NavbarSimple(
        brand="Domain Randomization",
        color="primary",
        dark=True,
        style={"margin-bottom": "20px"}
    ),

    # Buttons for "Mit Shift" and "Ohne Shift"
    html.Div([
        dbc.Button("Mit Shift", href="/mit_shift", color="success", className="me-2"),  # Changed to success theme
        dbc.Button("Ohne Shift", href="/ohne_shift", color="success")  # Changed to danger theme
    ], style={"position": "absolute", "top": "10px", "right": "10px"}),

    # Space for the content of the pages
    dash.page_container
])

# Define the default page
@app.callback(
    dash.dependencies.Output('url', 'pathname'),
    dash.dependencies.Input('url', 'pathname')
)
def redirect_to_default(pathname):
    if pathname == "/":
        return "/ohne_shift"  # Redirect to the default page
    return pathname

if __name__ == "__main__":
    app.run_server(debug=True)