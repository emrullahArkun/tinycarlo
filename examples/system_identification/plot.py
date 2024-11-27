from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Actor, Critic, and Reward Loss Graphs'),
    html.Div([
        dcc.Dropdown(
            id='domain_selector',
            options=[
                {'label': 'With Domain Randomization', 'value': 'with'},
                {'label': 'Without Domain Randomization', 'value': 'without'}
            ],
            value='without',  # Default selection
            style={"width": "50%"}
        )
    ], style={"margin-bottom": "20px"}),  # Space between dropdown and graphs

    html.Div(
        id="graph_container",  # Container for the graphs
        style={"display": "flex", "flex-direction": "row"}  # Side-by-side arrangement
    ),
    html.Div(id="reward_sum", style={"margin-top": "20px"})  # Container for the sum of rewards
])


@app.callback(
    [Output("graph_container", "children"),
     Output("reward_sum", "children")],
    Input("domain_selector", "value")
)
def update_page(domain_option):
    if domain_option == "with":
        # "With Domain Randomization" shows nothing
        return [], ""

    elif domain_option == "without":
        # Load CSV files
        actor_file = '/home/emrullah/Schreibtisch/actor_loss.csv'
        critic_file = '/home/emrullah/Schreibtisch/critic_loss.csv'
        reward_file = '/home/emrullah/Schreibtisch/rew.csv'

        actor_df = pd.read_csv(actor_file)
        critic_df = pd.read_csv(critic_file)
        reward_df = pd.read_csv(reward_file)

        # Calculate rolling mean
        actor_df['Loss'] = actor_df['Loss'].rolling(window=200).mean()
        critic_df['Critic 1'] = critic_df['Critic 1 Loss'].rolling(window=400).mean()
        critic_df['Critic 2'] = critic_df['Critic 2 Loss'].rolling(window=400).mean()

        # Calculate sum of rewards
        total_reward = round(reward_df['Reward'].sum(), 2)

        # Actor Loss Graph
        actor_fig = px.line(actor_df, x='Step', y='Loss', title="Actor Loss")
        actor_fig.update_layout(
            autosize=False,
            width=500,
            height=500
        )

        critic_fig = px.line(
            critic_df,
            x='Step',
            y=['Critic 1', 'Critic 2'],
            title="Critic Loss",
            labels={"value": "Loss"}
        )
        critic_fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            title_x=0.2,  # Center the title horizontally
            title_y=0.9,  # Adjust the vertical position of the title
            legend_title_text='',  # Entfernt den Titel der Legende
            legend=dict(
                orientation="h",  # Horizontale Ausrichtung
                yanchor="bottom",
                y=1.02,  # Platzierung knapp oberhalb des Diagramms
                xanchor="center",  # Zentrierte Ausrichtung
                x=0.35  # Verschiebt die Legende nach links oder rechts; 0.5 ist mittig, kleiner = weiter links
            )
        )

        # Reward Graph
        reward_fig = px.line(reward_df, x='Episode', y='Reward', title="Reward")
        reward_fig.update_layout(
            autosize=False,
            width=500,
            height=500
        )

        # Return graphs as HTML components and sum of rewards
        return [
            dcc.Graph(
                id="actor_graph",
                figure=actor_fig,
                style={"width": "500px", "height": "500px"}  # Square size for Actor Loss
            ),
            dcc.Graph(
                id="critic_graph",
                figure=critic_fig,
                style={"width": "500px", "height": "500px"}  # Square size for Critic Loss
            ),
            dcc.Graph(
                id="reward_graph",
                figure=reward_fig,
                style={"width": "500px", "height": "500px"}  # Square size for Reward
            )
        ], f"Total Reward: {total_reward}"


if __name__ == "__main__":
    app.run_server(debug=True)