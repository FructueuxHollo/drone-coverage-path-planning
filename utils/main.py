# main.py

import shapely.geometry as sg
import matplotlib.pyplot as plt
from planner import STCPlanner, STCVisualizer

def main():
    print("Initialisation du planificateur de patrouille STC...")
    
    # --- Étape 1: Définition de l'Environnement ---
    
    # Taille de l'outil du drone (ex: 10 mètres de couverture)
    TOOL_SIZE = 10

    # Zone de travail (un polygone plus complexe)
    # work_area = sg.Polygon([
    #     (0, 0), (0, 20), (40, 20), (40, 60), (60, 60), 
    #     (60, 20), (100, 20), (100, 0), (0, 0)
    # ])
    work_area = sg.Polygon([
        (0, 0), (200, 0), (200, 150), (100, 150), (100, 80), 
        (50, 80), (50, 150), (0, 150), (0, 0)
    ])

    # Obstacles
    obstacles = []
    obstacles = [
        sg.box(30, 40, 70, 60),
        sg.Polygon([(120, 50), (140, 50), (130, 90)])
    ]

    # Point de départ du drone
    start_point = sg.Point(10, 10)

    # --- Initialisation des modules ---
    planner = STCPlanner(work_area, obstacles, TOOL_SIZE)
    visualizer = STCVisualizer(work_area, obstacles)

    # --- Lancement de la planification ---
    # La méthode plan() va orchestrer les étapes et les appels au visualizer
    final_path_polygons, animation_obj = planner.plan(start_point, visualizer)
    
    if final_path_polygons:
        print(f"\nProcessus terminé. {len(final_path_polygons)} waypoints générés.")
        print("Fermez la fenêtre de visualisation pour terminer le programme.")
        plt.show() # Bloquant pour garder la fenêtre ouverte
    else:
        print("\nLa planification a échoué. Aucun chemin n'a été généré.")

if __name__ == '__main__':
    main()