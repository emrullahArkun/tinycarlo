import csv
from typing import List

import imageio.v2 as imageio  # GIF direkt während der Laufzeit erstellen
import numpy as np


def visualize_observation_streaming(writer, observation: np.ndarray):
    import matplotlib.pyplot as plt
    # Temporärer Pfad für die Zwischenbilder
    TEMP_PATH = '/tmp/observation.png'
    plt.imshow(observation)
    plt.axis('off')  # Achsen ausblenden
    plt.savefig(TEMP_PATH)  # Temporäre Datei speichern
    plt.close()  # Speicher freigeben
    image = imageio.imread(TEMP_PATH)
    writer.append_data(image)  # Bild direkt in das GIF schreiben

def create_distance_graph(laneline_distances: List[float], graph_name: str, shift_suffix: str):
    # CSV speichern
    with open(f"domain_randomization/data/{graph_name}_{shift_suffix}", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Distance"])  # Header
        for step, dist in enumerate(laneline_distances, start=1):
            writer.writerow([step, dist])

