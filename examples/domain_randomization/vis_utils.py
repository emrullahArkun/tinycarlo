import csv
from typing import List

import imageio.v2 as imageio  # GIF direkt während der Laufzeit erstellen
import numpy as np
from matplotlib import pyplot as plt


def create_distance_graph(laneline_distances: List[float], graph_name: str, shift_suffix: str):
    # CSV speichern
    with open(f"domain_randomization/data/{graph_name}_{shift_suffix}", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Distance"])  # Header
        for step, dist in enumerate(laneline_distances, start=1):
            writer.writerow([step, dist])

def create_cte_graph(cte_values: List[float], graph_name: str, shift_suffix: str):
    # CSV speichern
    with open(f"domain_randomization/data/{graph_name}_{shift_suffix}", mode="w", newline="") as file:
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
    csv_filename = f"domain_randomization/data/{network}_weight_changes_{shift_suffix}"

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Layer", "Mean Absolute Weight Change"])  # Header

        for step in steps:
            for layer_name, changes in weight_changes[step].items():
                mean_change = np.mean(changes)
                writer.writerow([step, layer_name, mean_change])