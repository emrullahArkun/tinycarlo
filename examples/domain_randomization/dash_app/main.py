from dash import Dash, dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import subprocess

# Use a Bootstrap theme
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CYBORG])

# Layout of the application
app.layout = html.Div([
    dcc.Location(id="url"),  # Monitors the URL of the application

    # Conditional Navbar
    html.Div(id="navbar-container"),

    # Placeholder for the shift dropdown
    html.Div(id="shift-dropdown", style={"position": "absolute", "top": "5px", "right": "5px"}),

    # Button to start the training script
    html.Button("Start Training", id="start-training-button", n_clicks=0),

    # Space for the content of the pages
    dash.page_container
])


# Callback to show/hide Navbar based on the current URL
@app.callback(
    Output("navbar-container", "children"),
    Input("url", "pathname")
)
def show_navbar(pathname):
    if pathname == "/":  # Home page path
        return None  # Do not render the Navbar
    else:
        return dbc.Navbar(
            color="purple",
            dark=True,
            style={"margin-bottom": "20px"},
            children=[
                dbc.NavItem(dbc.Button("Lenkwinkel", id="lenkwinkel-button", href="/lenkwinkel/ohne_shift", color="light",
                                       className="me-2",
                                       style={"font-size": "18px", "border-radius": "20px", "padding": "10px 20px",
                                              "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "margin-left": "5px"})),
                dbc.NavItem(dbc.Button("Akku", id="akku-button", href="/akku/ohne_shift", color="light",
                                       style={"font-size": "18px", "border-radius": "20px", "padding": "10px 20px",
                                              "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "margin-left": "5px"}))
            ]
        )


# Fix: Callback wird beim Seitenladen und Seitenwechsel getriggert
@app.callback(
    Output("shift-dropdown", "children"),
    Input("url", "pathname")
)
def update_shift_dropdown(pathname):
    shift_label = "Mit Shift" if "mit_shift" in pathname else "Ohne Shift"

    if pathname.startswith("/lenkwinkel"):
        return dbc.DropdownMenu(
            label=shift_label,
            children=[
                dbc.DropdownMenuItem("Mit Shift", href="/lenkwinkel/mit_shift"),
                dbc.DropdownMenuItem("Ohne Shift", href="/lenkwinkel/ohne_shift")
            ],
            color="primary",
            className="me-2",
            style={"border-radius": "20px", "padding": "10px 20px", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}
        )
    elif pathname.startswith("/akku"):
        return dbc.DropdownMenu(
            label=shift_label,
            children=[
                dbc.DropdownMenuItem("Mit Shift", href="/akku/mit_shift"),
                dbc.DropdownMenuItem("Ohne Shift", href="/akku/ohne_shift")
            ],
            color="success",
            className="me-2",
            style={"border-radius": "20px", "padding": "10px 20px", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}
        )
    return None  # Keine Dropdowns für Home oder ungültige Pfade


# Callback to start the training script
@app.callback(
    Output("start-training-button", "n_clicks"),
    Input("start-training-button", "n_clicks")
)
def start_training(n_clicks):
    if n_clicks > 0:
        subprocess.Popen(["python", "/home/emrullah/Schreibtisch/tinycarlo/examples/train_td3.py"])

    return 0  # Reset the button clicks


if __name__ == "__main__":
    app.run_server(debug=True)