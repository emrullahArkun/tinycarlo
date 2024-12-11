from dash import Dash, dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc

# Use a Bootstrap theme
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CYBORG])

# Layout of the application
app.layout = html.Div([
    dcc.Location(id="url"),  # Monitors the URL of the application

    # Navigation bar with links to the pages
    dbc.Navbar(
        color="primary",
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
    ),

    # Placeholder for the shift buttons
    html.Div(id="shift-buttons", style={"position": "absolute", "top": "10px", "right": "10px"}),

    # Space for the content of the pages
    dash.page_container
])

# Callback to update the button color based on the click event and current page
@app.callback(
    [Output("lenkwinkel-button", "color"), Output("akku-button", "color"),
     Output("shift-buttons", "children")],
    [Input("lenkwinkel-button", "n_clicks"), Input("akku-button", "n_clicks"),
     Input("url", "pathname")],
    [State("lenkwinkel-button", "color"), State("akku-button", "color")]
)
def update_button_color(lenkwinkel_clicks, akku_clicks, pathname, lenkwinkel_color, akku_color):
    ctx = dash.callback_context

    if not ctx.triggered:
        return lenkwinkel_color, akku_color, []

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if pathname.startswith("/lenkwinkel"):
        lenkwinkel_color = "light"
        akku_color = "light"
        shift_buttons = [
            dbc.Button("Mit Shift", id="mit-shift-button", href="/lenkwinkel/mit_shift", color="success" if button_id != "mit-shift-button" else "danger", className="me-2", style={"border-radius": "20px", "padding": "10px 20px", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}),
            dbc.Button("Ohne Shift", id="ohne-shift-button", href="/lenkwinkel/ohne_shift", color="success" if button_id != "ohne-shift-button" else "danger", style={"border-radius": "20px", "padding": "10px 20px", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
        ]
    elif pathname.startswith("/akku"):
        lenkwinkel_color = "light"
        akku_color = "light"
        shift_buttons = [
            dbc.Button("Mit Shift", id="mit-shift-button", href="/akku/mit_shift", color="success" if button_id != "mit-shift-button" else "danger", className="me-2", style={"border-radius": "20px", "padding": "10px 20px", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}),
            dbc.Button("Ohne Shift", id="ohne-shift-button", href="/akku/ohne_shift", color="success" if button_id != "ohne-shift-button" else "danger", style={"border-radius": "20px", "padding": "10px 20px", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
        ]
    else:
        shift_buttons = []

    return lenkwinkel_color, akku_color, shift_buttons

if __name__ == "__main__":
    app.run_server(debug=True)