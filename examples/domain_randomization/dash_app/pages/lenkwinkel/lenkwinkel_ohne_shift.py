# lenkwinkel_ohne_shift.py
import os
from dash import register_page

from examples.domain_randomization.dash_app.pages.layout_utils import create_layout

register_page(__name__, path="/lenkwinkel/ohne_shift")

base_path = '../data'

def layout():
    return create_layout(
        'Ohne Shift im Lenkwinkel',
        os.path.join(base_path, 'outer_without_shift'),
        os.path.join(base_path, 'dashed_without_shift'),
        os.path.join(base_path, 'solid_without_shift'),
        os.path.join(base_path, 'actor_weight_changes_without_shift'),
        os.path.join(base_path, 'critic1_weight_changes_without_shift'),
        os.path.join(base_path, 'critic2_weight_changes_without_shift'),
        os.path.join(base_path, 'critic_loss_without_shift'),
        os.path.join(base_path, 'actor_loss_without_shift'),
        os.path.join(base_path, 'rew_without_shift'),
        os.path.join(base_path, 'cte_without_shift'),
        os.path.join(base_path, 'latent_space_visualization_without_shift')
    )