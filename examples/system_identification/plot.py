from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Actor und Critic Loss Graphen'),
    html.Div([
        # Actor Loss Graph
        dcc.Graph(
            id="actor_graph",
            style={"width": "1000px", "height": "1000px"}  # Quadratische Größe für Actor Loss
        ),
        # Critic Loss Graph
        dcc.Graph(
            id="critic_graph",
            style={"width": "1100px", "height": "1000px"}  # Quadratische Größe für Critic Loss
        )
    ], style={"display": "flex", "flex-direction": "row"})  # Nebeneinander anzeigen
])


@app.callback(
    [Output("actor_graph", "figure"),
     Output("critic_graph", "figure")],
    [Input("actor_graph", "id"), Input("critic_graph", "id")]
)
def display_graph(_, __):
    # CSV-Datei für Actor Loss lesen
    actor_df = pd.read_csv('/home/emrullah/Schreibtisch/actor_loss.csv')

    # CSV-Datei für Critic Loss lesen
    critic_df = pd.read_csv('/home/emrullah/Schreibtisch/critic_loss.csv')

    # Gleitender Durchschnitt berechnen für Actor Loss
    actor_df['Loss'] = actor_df['Loss'].rolling(window=200).mean()

    # Gleitender Durchschnitt berechnen für beide Critic Losses
    critic_df['Critic_1_Loss'] = critic_df['Critic 1 Loss'].rolling(window=200).mean()
    critic_df['Critic_2_Loss'] = critic_df['Critic 2 Loss'].rolling(window=200).mean()

    # Plotly-Grafik für Actor Loss erstellen
    actor_fig = px.line(actor_df, x='Step', y='Loss', title="Actor Loss")
    actor_fig.update_layout(
        autosize=False,
        width=1000,  # Feste Breite
        height=1000  # Feste Höhe
    )

    # Plotly-Grafik für Critic Loss erstellen (mit beiden smoothed Critic Losses)
    critic_fig = px.line(
        critic_df,
        x='Step',
        y=['Critic_1_Loss', 'Critic_2_Loss'],
        title="Critic Loss"
    )
    critic_fig.update_layout(
        autosize=False,
        width=1100,  # Feste Breite
        height=1000  # Feste Höhe
    )

    return actor_fig, critic_fig


if __name__ == "__main__":
    app.run_server(debug=True)
