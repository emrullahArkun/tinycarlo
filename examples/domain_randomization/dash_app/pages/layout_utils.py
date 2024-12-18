# layout_utils.py
import plotly.io as pio
from dash import html, dcc
import plotly.express as px
import pandas as pd
import dash

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
def create_layout(title, outer_file, dashed_file, solid_file, actor_weight_file, critic1_weight_file, critic2_weight_file, critic_loss_file, actor_loss_file, ep_rew_file, cte_file, latent_space):
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
    outer_df['New Distance'] = outer_df['Distance'].rolling(window=window_size).mean()
    dashed_df['New Distance'] = dashed_df['Distance'].rolling(window=window_size).mean()
    solid_df['New Distance'] = solid_df['Distance'].rolling(window=window_size).mean()
    actor_loss['New Loss'] = actor_loss['Loss'].rolling(window=window_size).mean()
    critic_loss['New Loss'] = critic_loss['Loss'].rolling(window=window_size).mean()
    cte['New CTE'] = cte['CTE'].rolling(window=window_size).mean()

    # Create the plots
    outer_fig = px.line(outer_df, x='New Distance', y='Step', title="Outer Distance", template="custom_template")
    dashed_fig = px.line(dashed_df, x='New Distance', y='Step', title="Dashed Distance", template="custom_template")
    solid_fig = px.line(solid_df, x='New Distance', y='Step', title="Solid Distance", template="custom_template")

    critic_loss_fig = px.line(critic_loss, x='Step', y='New Loss', color='Critic', title="Critic Loss", template="custom_template")
    actor_loss_fig = px.line(actor_loss, x='Step', y='New Loss', title="Actor Loss", template="custom_template")
    ep_rew_fig = px.line(ep_rew, x='Episode', y='Reward', title="Episodic Reward", template="custom_template")
    cte_fig = px.line(cte, x='Step', y='New CTE', title="CTE", template="custom_template")

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

    # Load the latent space CSV file
    latent_df = pd.read_csv(latent_space)

    # Plot 1: Reward-Visualisierung
    reward_ls_fig = px.scatter(
        latent_df, x="Dimension1", y="Dimension2", color="Reward",
        title="Latent Space Visualization (Reward)",
        template="custom_template",
        color_continuous_scale="viridis", hover_data=["Reward", "CTE", "Maneuver"]
    )

    # Plot 2: CTE-Visualisierung
    cte_ls_fig = px.scatter(
        latent_df, x="Dimension1", y="Dimension2", color="CTE",
        title="Latent Space Visualization (CTE)",
        template="custom_template",
        color_continuous_scale="rdbu", hover_data=["Reward", "CTE", "Maneuver"]
    )

    # Plot 3: Maneuver-Visualisierung
    maneuver_ls_fig = px.scatter(
        latent_df, x="Dimension1", y="Dimension2", color="Maneuver",
        title="Latent Space Visualization (Maneuver)",
        template="custom_template",
        color_discrete_map={0: "red", 1: "green", 2: "blue"}, hover_data=["Reward", "CTE", "Maneuver"]
    )

    # Adjust layout for the plots
    for fig in [outer_fig, dashed_fig, solid_fig, critic_loss_fig, actor_loss_fig, ep_rew_fig, cte_fig, actor_weight_fig, critic1_weight_fig, critic2_weight_fig, reward_ls_fig, cte_ls_fig, maneuver_ls_fig]:
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            title_x=0.5,
            title_y=0.85
        )

    # Update y-axis line color and position for outer_fig
    outer_fig.update_yaxes(linecolor='red', side='right')

    # Add vertical dashed yellow line for dashed_fig
    dashed_fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=dashed_df['Step'].min(),
        y1=dashed_df['Step'].max(),
        line=dict(
            color="yellow",
            width=2,
            dash="dash"
        )
    )

    # Update y-axis line color and position for solid_fig
    solid_fig.update_yaxes(linecolor='red', side='left')

    # Return the layout of the page
    return html.Div([
        html.H1(title, style={
            "text-align": "center", "font-size": "60px", "color": "#ffffff", "font-family": "'Courier New', monospace",
            "text-shadow": "2px 2px 0px #000000", "margin-bottom": "30px", }),
        html.H3("Bild / Segmentation"),
        html.Div([
            dcc.Graph(figure=outer_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=dashed_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=solid_fig, style={"width": "500px", "height": "500px"})
        ], style={"display": "flex", "flex-direction": "row", "flex-wrap": "wrap"}),
        html.H3("Encoder"),
        html.Div([
            dcc.Graph(figure=reward_ls_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=cte_ls_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=maneuver_ls_fig, style={"width": "500px", "height": "500px"})
        ], style={"display": "flex", "flex-direction": "row", "flex-wrap": "wrap"}),
        html.H3("RL-Netze"),
        html.Div([
            dcc.Graph(figure=actor_loss_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=critic_loss_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=ep_rew_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=cte_fig, style={"width": "500px", "height": "500px"})
        ], style={"display": "flex", "flex-direction": "row", "flex-wrap": "wrap"}),
        html.Div([
            dcc.Graph(figure=actor_weight_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=critic1_weight_fig, style={"width": "500px", "height": "500px"}),
            dcc.Graph(figure=critic2_weight_fig, style={"width": "500px", "height": "500px"})
        ], style={"display": "flex", "flex-direction": "row", "flex-wrap": "wrap"}),
    ])