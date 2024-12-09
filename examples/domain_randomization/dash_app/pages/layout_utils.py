# layout_utils.py
import plotly.io as pio
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Define a figure template
pio.templates["custom_template"] = pio.templates["simple_white"]
pio.templates["custom_template"].layout.update(
    {
        "font": {"family": "Arial", "size": 12, "color": "white"},
        "title": {"x": 0.5, "xanchor": "center"},
        "paper_bgcolor": "#060606",
        "plot_bgcolor": "#060606",
    }
)

# Apply the template to your figures
def create_layout(title, outer_file, dashed_file, solid_file, actor_weight_file, critic1_weight_file, critic2_weight_file, critic_loss_file, actor_loss_file, ep_rew_file, cte_file):
    # Load the CSV files
    outer_df = pd.read_csv(outer_file)
    dashed_df = pd.read_csv(dashed_file)
    solid_df = pd.read_csv(solid_file)
    actor_weight_changes = pd.read_csv(actor_weight_file)
    critic1_weight_changes = pd.read_csv(critic1_weight_file)
    critic2_weight_changes = pd.read_csv(critic2_weight_file)

    critic_loss = pd.read_csv(critic_loss_file)
    critic_loss = critic_loss.melt(id_vars=['Step'], value_vars=['Critic 1 Loss', 'Critic 2 Loss'], var_name='Critic', value_name='Loss')
    actor_loss = pd.read_csv(actor_loss_file)
    ep_rew = pd.read_csv(ep_rew_file)
    cte = pd.read_csv(cte_file)

    # Calculate rolling mean for the data
    window_size = 500
    outer_df['Distance'] = outer_df['Distance'].rolling(window=window_size).mean()
    dashed_df['Distance'] = dashed_df['Distance'].rolling(window=window_size).mean()
    solid_df['Distance'] = solid_df['Distance'].rolling(window=window_size).mean()
    actor_loss['Loss'] = actor_loss['Loss'].rolling(window=window_size).mean()
    critic_loss['Loss'] = critic_loss['Loss'].rolling(window=window_size).mean()
    cte['CTE'] = cte['CTE'].rolling(window=window_size).mean()

    # Create the plots
    outer_fig = px.line(outer_df, x='Step', y='Distance', title="Outer Distance", template="custom_template")
    dashed_fig = px.line(dashed_df, x='Step', y='Distance', title="Dashed Distance", template="custom_template")
    solid_fig = px.line(solid_df, x='Step', y='Distance', title="Solid Distance", template="custom_template")

    critic_loss_fig = px.line(critic_loss, x='Step', y='Loss', color='Critic', title="Critic Loss", template="custom_template")
    actor_loss_fig = px.line(actor_loss, x='Step', y='Loss', title="Actor Loss", template="custom_template")
    ep_rew_fig = px.line(ep_rew, x='Episode', y='Reward', title="Episodic Reward", template="custom_template")
    cte_fig = px.line(cte, x='Step', y='CTE', title="CTE", template="custom_template")

    # Assign different colors to each type of weight
    color_discrete_map = {
        'cnn1.weight': 'red',
        'cnn2.weight': 'red',
        'cnn3.weight': 'red',
        'fcm1.weight': 'blue',
        'fcm2.weight': 'blue',
        'fcm3.weight': 'blue',
        'fc1.weight': 'green',
        'fc2.weight': 'green',
        'fc3.weight': 'green',
        'fc4.weight': 'green'
    }
    actor_weight_fig = px.line(actor_weight_changes, x='Step', y='Mean Absolute Weight Change', color='Layer', title="Actor Weight Changes", template="custom_template", color_discrete_map=color_discrete_map)
    critic1_weight_fig = px.line(critic1_weight_changes, x='Step', y='Mean Absolute Weight Change', color='Layer', title="Critic1 Weight Changes", template="custom_template", color_discrete_map=color_discrete_map)
    critic2_weight_fig = px.line(critic2_weight_changes, x='Step', y='Mean Absolute Weight Change', color='Layer', title="Critic2 Weight Changes", template="custom_template", color_discrete_map=color_discrete_map)

    # Adjust layout for the plots
    for fig in [outer_fig, dashed_fig, solid_fig, critic_loss_fig, actor_loss_fig, ep_rew_fig, cte_fig, actor_weight_fig, critic1_weight_fig, critic2_weight_fig]:
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            title_x=0.5,
            title_y=0.85
        )

    # Return the layout of the page
    return html.Div([
        html.H1(title),
        html.H3("Bild / Segmentation"),
        html.Div([
            dcc.Graph(figure=outer_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=dashed_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=solid_fig, style={"width": "500px", "height": "500px"})
        ], style={"display": "flex", "flex-direction": "row", "flex-wrap": "wrap"}),
        html.H3("Encoder"),
        html.H3("RL-Netze"),
        html.Div([
            dcc.Graph(figure=actor_loss_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=critic_loss_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=ep_rew_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=cte_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=actor_weight_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=critic1_weight_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=critic2_weight_fig, style={"width": "500px", "height": "500px"})
        ], style={"display": "flex", "flex-direction": "row", "flex-wrap": "wrap"}),
    ])