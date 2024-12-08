import csv
from typing import List

import imageio.v2 as imageio  # GIF direkt während der Laufzeit erstellen
import numpy as np


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
