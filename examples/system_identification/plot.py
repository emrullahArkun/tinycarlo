from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Actor und Critic Loss Graphen'),
    html.Div([
        dcc.Dropdown(
            id='domain_selector',
            options=[
                {'label': 'With Domain Randomization', 'value': 'with'},
                {'label': 'Without Domain Randomization', 'value': 'without'}
            ],
            value='without',  # Standardauswahl
            style={"width": "50%"}
        )
    ], style={"margin-bottom": "20px"}),  # Abstand zwischen Dropdown und Graphen

    html.Div(
        id="graph_container",  # Container für die Graphen
        style={"display": "flex", "flex-direction": "row"}  # Nebeneinander-Anordnung
    )
])


@app.callback(
    Output("graph_container", "children"),
    Input("domain_selector", "value")
)
def update_page(domain_option):
    if domain_option == "with":
        # "With Domain Randomization" zeigt nichts an
        return []

    elif domain_option == "without":
        # CSV-Dateien laden
        actor_file = '/home/emrullah/Schreibtisch/actor_loss.csv'
        critic_file = '/home/emrullah/Schreibtisch/critic_loss.csv'

        actor_df = pd.read_csv(actor_file)
        critic_df = pd.read_csv(critic_file)

        # Gleitender Durchschnitt berechnen
        actor_df['Loss'] = actor_df['Loss'].rolling(window=10).mean()
        critic_df['Critic_1_Loss'] = critic_df['Critic 1 Loss'].rolling(window=20).mean()
        critic_df['Critic_2_Loss'] = critic_df['Critic 2 Loss'].rolling(window=20).mean()

        # Actor Loss Graph
        actor_fig = px.line(actor_df, x='Step', y='Loss', title="Actor Loss (Without Domain Randomization)")
        actor_fig.update_layout(
            autosize=False,
            width=1000,
            height=1000
        )

        # Critic Loss Graph
        critic_fig = px.line(
            critic_df,
            x='Step',
            y=['Critic_1_Loss', 'Critic_2_Loss'],
            title="Critic Loss (Without Domain Randomization)"
        )
        critic_fig.update_layout(
            autosize=False,
            width=1100,
            height=1000
        )

        # Graphen als HTML-Komponenten zurückgeben
        return [
            dcc.Graph(
                id="actor_graph",
                figure=actor_fig,
                style={"width": "1000px", "height": "1000px"}  # Quadratische Größe für Actor Loss
            ),
            dcc.Graph(
                id="critic_graph",
                figure=critic_fig,
                style={"width": "1100px", "height": "1000px"}  # Quadratische Größe für Critic Loss
            )
        ]


if __name__ == "__main__":
    app.run_server(debug=True)
