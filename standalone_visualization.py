# --- Fichier : run_standalone_visualization.py ---

# On importe les classes de nos autres fichiers
from stc_path_planner import DronePathPlanner
from stc_visualizer import MissionVisualizer
import matplotlib.pyplot as plt

def main():
    print("--- Lancement du planificateur en mode standalone avec visualisation pas à pas ---")
    
    # 1. Définition de la mission (vous pouvez modifier ces paramètres)
    FLIGHT_ALTITUDE_M = 43
    CAMERA_FOV_DEGREES = 60
    OVERLAP_PERCENTAGE = 0
    
    zone = [
        (0, 0), (1000, 0), (1000, 750), (500, 750), (500, 400), 
        (250, 400), (250, 750), (0, 750)
    ]
    obstacles = [
        [(150, 200), (350, 200), (350, 300), (150, 300)],
        # [(120, 50), (140, 50), (130, 90)]
        [(600, 250), (700, 250), (650, 300)]
    ]
    start_coords = (200, 500) # Point de départ en (East, North)

    # 2. Initialisation du planificateur
    try:
        planner = DronePathPlanner(
            polygon_vertices=zone,
            altitude_m=FLIGHT_ALTITUDE_M,
            camera_fov_degrees=CAMERA_FOV_DEGREES,
            obstacle_polygons=obstacles,
            start_point_coords=start_coords,
            overlap_percentage=OVERLAP_PERCENTAGE
        )
    except Exception as e:
        print(f"\nErreur lors de l'initialisation du planificateur : {e}")
        return

    # 3. Initialisation du visualiseur
    visualizer = MissionVisualizer(planner)

    # 4. Exécution et visualisation étape par étape
    
    # Étape 1: Afficher l'environnement initial
    visualizer.draw_step_1_environment()
    input("Appuyez sur Entrée pour passer à l'étape 2 (Grille)...")

    # Étape 2: Afficher la grille
    visualizer.draw_step_2_grid()
    input("Appuyez sur Entrée pour passer à l'étape 3 (Calcul et affichage de l'arbre)...")
    
    # Lancer la planification pour calculer l'arbre et le chemin
    final_path = planner.plan_coverage_path()

    if not final_path:
        print("\nLa planification a échoué. Arrêt de la visualisation.")
        return

    # Étape 3: Afficher l'arbre
    visualizer.draw_step_3_spanning_tree()
    input("Appuyez sur Entrée pour passer à l'étape 4 (Trajectoire finale)...")

    # Étape 4: Afficher la trajectoire finale
    visualizer.draw_step_4_final_path()
    
    # Garder la fenêtre ouverte jusqu'à ce que l'utilisateur la ferme
    visualizer.wait_for_close()

if __name__ == '__main__':
    main()