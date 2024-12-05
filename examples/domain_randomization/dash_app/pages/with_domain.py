from dash import html, register_page, dcc
import plotly.express as px
import pandas as pd

register_page(__name__, path="/with_shift")  # Registriert diese Datei als eine Seite

def layout():
    # Lade die CSV-Dateien
    outer_df = pd.read_csv('../data/outer_with_shift')
    dashed_df = pd.read_csv('../data/dashed_with_shift')
    solid_df = pd.read_csv('../data/solid_with_shift')
    hold_df = pd.read_csv('../data/hold_with_shift')
    area_df = pd.read_csv('../data/area_with_shift')

    critic_loss = pd.read_csv('../data/critic_loss_with_shift')
    critic_loss = critic_loss.melt(id_vars=['Step'], value_vars=['Critic 1 Loss', 'Critic 2 Loss'], var_name='Critic',value_name='Loss')
    actor_loss = pd.read_csv('../data/actor_loss_with_shift')
    ep_rew = pd.read_csv('../data/rew_with_shift')

    # Gleitender Durchschnitt für die Daten berechnen
    window_size = 200
    outer_df['Distance'] = outer_df['Distance'].rolling(window=window_size).mean()
    dashed_df['Distance'] = dashed_df['Distance'].rolling(window=window_size).mean()
    solid_df['Distance'] = solid_df['Distance'].rolling(window=window_size).mean()
    hold_df['Distance'] = hold_df['Distance'].rolling(window=window_size).mean()
    area_df['Distance'] = area_df['Distance'].rolling(window=window_size).mean()

    # Erstelle die Diagramme
    outer_fig = px.line(outer_df, x='Step', y='Distance', title="Outer Distance")
    dashed_fig = px.line(dashed_df, x='Step', y='Distance', title="Dashed Distance")
    solid_fig = px.line(solid_df, x='Step', y='Distance', title="Solid Distance")
    hold_fig = px.line(hold_df, x='Step', y='Distance', title="Hold Distance")
    area_fig = px.line(area_df, x='Step', y='Distance', title="Area Distance")

    critic_loss_fig = px.line(critic_loss, x='Step', y='Loss', color='Critic', title="Critic Loss")
    actor_loss_fig = px.line(actor_loss, x='Step', y='Loss', title="Actor Loss")
    ep_rew_fig = px.line(ep_rew, x='Episode', y='Reward', title="Episodic Reward")

    # Layout für die Diagramme anpassen
    for fig in [outer_fig, dashed_fig, solid_fig, hold_fig, area_fig, critic_loss_fig, actor_loss_fig, ep_rew_fig]:
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            title_x=0.5,
            title_y=0.85
        )

    # Rückgabe des Layouts der Seite
    return html.Div([
        html.H1("Shift im Lenkwinkel"),
        html.H3("Bild / Segmentation"),
        html.Div([
            dcc.Graph(figure=outer_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=dashed_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=solid_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=hold_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=area_fig, style={"width": "500px", "height": "500px"})
        ], style={"display": "flex", "flex-direction": "row", "flex-wrap": "wrap"}),
        html.H3("Encoder"),
        html.H3("RL-Netze"),
        html.Div([
            dcc.Graph(figure=actor_loss_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=critic_loss_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=ep_rew_fig, style={"width": "500px", "height": "500px"}),
        ], style={"display": "flex", "flex-direction": "row", "flex-wrap": "wrap"}),
    ])
