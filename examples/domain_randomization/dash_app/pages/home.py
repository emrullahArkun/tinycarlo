from dash import html, register_page
import dash_bootstrap_components as dbc

register_page(__name__, path='/')

layout = html.Div([
    html.H1("Analyse-Tool für den Tinycar", style={
        "text-align": "center", "font-size": "80px", "color": "#ffffff", "font-family": "'Courier New', monospace",
        "text-shadow": "2px 2px 0px #000000", "margin-top": "70px", "margin-bottom": "50px", }),

    # Große Box
    html.Div([
        # Kleine Box
        html.Div([
            html.H2("Lenkwinkel", style={
                "text-align": "center", "font-size": "40px", "color": "#ffffff",
                "font-family": "'Courier New', monospace",
                "text-shadow": "2px 2px 0px #000000", "margin-top": "20px", "margin-bottom": "20px", }),

            # Buttons in einer Reihe
            html.Div([
                dbc.Button("Mit Shift", href="/lenkwinkel/mit_shift", color="success", className="me-2",
                           style={"font-size": "24px", "border-radius": "20px", "padding": "15px 30px",
                                  "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "margin": "20px"}),  # Grün
                dbc.Button("Ohne Shift", href="/lenkwinkel/ohne_shift", color="danger", className="me-2",
                           style={"font-size": "24px", "border-radius": "20px", "padding": "15px 30px",
                                  "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "margin": "20px"})  # Rot
            ], style={"display": "flex", "justify-content": "center", "gap": "20px"})  # Flexbox für horizontale Anordnung
        ], style={
            "text-align": "center", "border": "2px solid #ffffff", "padding": "30px", "border-radius": "10px",
            "background-color": "#333333", "width": "400px", "height": "300px", "position": "absolute",
            "top": "50px", "left": "50px"  # Abstand von oben und links vergrößert
        })
    ], style={
        "border": "2px solid #ffffff", "border-radius": "15px", "background-color": "#222222",
        "width": "95%", "height": "600px", "margin": "0 auto", "position": "relative", "padding": "30px"
    })
])