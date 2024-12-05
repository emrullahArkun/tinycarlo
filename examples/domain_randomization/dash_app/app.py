from dash import Dash, dcc, html
import dash

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)  # Aktiviert Seiten und unterdrückt Fehler für dynamische IDs

# Layout der Anwendung
app.layout = html.Div([
    dcc.Location(id="url"),  # Überwacht die URL der Anwendung

    # Navigationsleiste mit Links zu den Seiten
    html.Div([
        dcc.Link("With Shift", href="/with_shift", style={"margin-right": "20px"}),
        dcc.Link("Without Shift", href="/without_shift", style={"margin-right": "20px"})
    ], style={"padding": "10px", "background-color": "#f2f2f2", "margin-bottom": "20px"}),


    # Platz für den Inhalt der Seiten
    dash.page_container
])

# Standardseite definieren
@app.callback(
    dash.dependencies.Output('url', 'pathname'),
    dash.dependencies.Input('url', 'pathname')
)
def redirect_to_default(pathname):
    if pathname == "/":
        return "/without_shift"  # Weiterleitung zur Standardseite
    return pathname

if __name__ == "__main__":
    app.run_server(debug=True)
