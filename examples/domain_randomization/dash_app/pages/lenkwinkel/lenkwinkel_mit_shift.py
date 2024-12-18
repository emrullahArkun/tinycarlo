import os
from dash import register_page

from examples.domain_randomization.dash_app.pages.layout_utils import create_layout

register_page(__name__, path="/lenkwinkel/mit_shift")

base_path = '/tmp'

def layout():
    return create_layout(
        'Mit Shift im Lenkwinkel',
        os.path.join(base_path, 'outer_with_shift'),
        os.path.join(base_path, 'dashed_with_shift'),
        os.path.join(base_path, 'solid_with_shift'),
        os.path.join(base_path, 'actor_weight_changes_with_shift'),
        os.path.join(base_path, 'critic1_weight_changes_with_shift'),
        os.path.join(base_path, 'critic2_weight_changes_with_shift'),
        os.path.join(base_path, 'critic_loss_with_shift'),
        os.path.join(base_path, 'actor_loss_with_shift'),
        os.path.join(base_path, 'rew_with_shift'),
        os.path.join(base_path, 'cte_with_shift'),
        os.path.join(base_path, 'latent_space_visualization_with_shift')
    )