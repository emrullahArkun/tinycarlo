import csv
import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.express as px
from dash import dcc, html


def create_distance_graph(laneline_distances: List[float], graph_name: str, shift_suffix: str):
    # CSV speichern
    #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./data/{graph_name}_{shift_suffix}")
    with open(f"/tmp/{graph_name}_{shift_suffix}", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Distance"])  # Header
        for step, dist in enumerate(laneline_distances, start=1):
            writer.writerow([step, dist])

def create_cte_graph(cte_values: List[float], graph_name: str, shift_suffix: str):
    # CSV speichern
    #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./data/{graph_name}_{shift_suffix}")
    with open(f"/tmp/{graph_name}_{shift_suffix}", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "CTE"])  # Header
        for step, cte in enumerate(cte_values, start=1):
            writer.writerow([step, cte])

def calculate_weight_changes(weight_history):
    weight_changes = {}
    steps = sorted(weight_history.keys())
    for i in range(1, len(steps)):
        step = steps[i]
        prev_step = steps[i - 1]
        weight_changes[step] = {}
        for name in weight_history[step]:
            weight_changes[step][name] = np.abs(weight_history[step][name] - weight_history[prev_step][name])
    return weight_changes


def plot_all_weight_changes(network, weight_changes, shift_suffix):
    steps = sorted(weight_changes.keys())
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./data/{network}_weight_changes_{shift_suffix}")

    with open(f"/tmp/{network}_weight_changes_{shift_suffix}", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Layer", "Absolute Weight Change"])  # Header

        for step in steps:
            for layer_name, changes in weight_changes[step].items():
                mean_change = np.mean(changes)
                writer.writerow([step, layer_name, mean_change])

def save_to_csv(latent_space_2d, rewards, ctes, maneuvers, heading_errors, shift_suffix):
    df = pd.DataFrame({
        "Dimension1": latent_space_2d[:, 0],
        "Dimension2": latent_space_2d[:, 1],
        "Reward": rewards,
        "CTE": ctes,
        "Maneuver": maneuvers,
        "Heading Error": heading_errors
    })
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"/tmp/latent_space_visualization_{shift_suffix}")
    df.to_csv(path, index=False)
