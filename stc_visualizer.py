import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class MissionVisualizer:
    """
    Classe dédiée à la visualisation des différentes étapes de l'algorithme de planification.
    """
    def __init__(self, planner_instance):
        self.planner = planner_instance
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Génération de la Trajectoire de Couverture", fontsize=16)

    def draw_step_1_environment(self):
        """Étape 1: Affiche la zone de travail et les obstacles."""
        print("\n--- Visualisation Étape 1: Environnement ---")
        
        # Dessiner la zone de travail
        zone_coords = self.planner.polygon_vertices.tolist()
        zone_coords.append(zone_coords[0]) # Fermer le polygone
        zone_xs, zone_ys = zip(*zone_coords)
        self.ax.plot(zone_xs, zone_ys, 'b-', linewidth=2, label='Zone de travail')

        # Dessiner les obstacles
        obstacle_label_added = False
        for obs_poly in self.planner.obstacle_polygons:
            label = 'Obstacle' if not obstacle_label_added else ""
            self.ax.add_patch(patches.Polygon(obs_poly, closed=True, facecolor='salmon', edgecolor='gray', label=label))
            obstacle_label_added = True

        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(r"D:\Fructueux\Work\Memoire\Drone Coverage Path Planning\Code\results\stc\zone_3_step_1.png")
        plt.show(block=False)
        plt.pause(0.1)

    def draw_step_2_grid(self):
        """Étape 2: Affiche la grille discrétisée."""
        print("\n--- Visualisation Étape 2: Grille Discrétisée ---")
        
        rows, cols = self.planner.grid.shape
        # Dessiner la grille (lignes vertes)
        for r in range(rows + 1):
            y = r * self.planner.resolution + self.planner.origin_offset[1]
            self.ax.axhline(y, color='green', linestyle='-', linewidth=0.5, alpha=0.4)
        for c in range(cols + 1):
            x = c * self.planner.resolution + self.planner.origin_offset[0]
            self.ax.axvline(x, color='green', linestyle='-', linewidth=0.5, alpha=0.4)
            
        self.ax.set_title("Étape 2: Grille Discrétisée", fontsize=16)
        plt.savefig(r"D:\Fructueux\Work\Memoire\Drone Coverage Path Planning\Code\results\stc\zone_3_step_2.png")
        plt.show(block=False)
        plt.pause(0.1)

    def draw_step_3_spanning_tree(self):
        """Étape 3: Affiche l'Arbre Recouvrant Minimum (MST)."""
        print("\n--- Visualisation Étape 3: Arbre Recouvrant Minimum ---")

        if not self.planner.path_from:
            print("Avertissement: Aucun arbre à visualiser.")
            return

        for point, from_point in self.planner.path_from.items():
            if from_point is not None:
                # Convertir les coordonnées de la grille (r, c) en coordonnées du monde (centre des cellules)
                start_r, start_c = from_point
                start_x = start_c * self.planner.resolution + self.planner.origin_offset[0] + self.planner.resolution / 2
                start_y = start_r * self.planner.resolution + self.planner.origin_offset[1] + self.planner.resolution / 2
                
                end_r, end_c = point
                end_x = end_c * self.planner.resolution + self.planner.origin_offset[0] + self.planner.resolution / 2
                end_y = end_r * self.planner.resolution + self.planner.origin_offset[1] + self.planner.resolution / 2
                
                self.ax.plot([start_x, end_x], [start_y, end_y], color='purple', linestyle='-', linewidth=2.5)

        self.ax.set_title("Étape 3: Arbre Recouvrant Minimum", fontsize=16)
        plt.savefig(r"D:\Fructueux\Work\Memoire\Drone Coverage Path Planning\Code\results\stc\zone_3_step_3.png")
        plt.show(block=False)
        plt.pause(0.1)

    def draw_step_4_final_path(self):
        """Étape 4: Affiche la trajectoire finale."""
        print("\n--- Visualisation Étape 4: Trajectoire Finale ---")

        if not self.planner.final_waypoints:
            print("Avertissement: Aucune trajectoire finale à visualiser.")
            return

        # 1. Dessiner la ligne de la trajectoire
        path_xs, path_ys = zip(*self.planner.final_waypoints)
        self.ax.plot(path_xs, path_ys, 'o-', color='darkorange', markersize=3, linewidth=1.5, label='Trajectoire du Drone')

        # 2. Ajouter des flèches directionnelles
        # On choisit de dessiner une flèche tous les 'arrow_spacing' points
        arrow_spacing = 30 
        
        # On extrait les coordonnées de la trajectoire dans un tableau numpy pour faciliter les calculs
        path_points = np.array(self.planner.final_waypoints)
        
        # On sélectionne les points de départ des flèches
        arrow_start_points = path_points[:-1:arrow_spacing]
        
        # On sélectionne les points d'arrivée correspondants
        arrow_end_points = path_points[1::arrow_spacing]
        
        # On calcule les vecteurs de direction (arrivée - départ)
        # U = dx (différence en x), V = dy (différence en y)
        U = arrow_end_points[:, 0] - arrow_start_points[:, 0]
        V = arrow_end_points[:, 1] - arrow_start_points[:, 1]
        
        # On extrait les coordonnées de départ des flèches
        X = arrow_start_points[:, 0]
        Y = arrow_start_points[:, 1]
        
        # On utilise 'quiver' pour dessiner toutes les flèches en un seul appel
        # scale_units='xy', scale=1 : pour que la longueur de la flèche corresponde à la distance entre les points
        # color='navy' : pour un bon contraste
        self.ax.quiver(X, Y, U, V, 
                       scale_units='xy', angles='xy', scale=1, 
                       color='navy', width=0.005, headwidth=3)

        self.ax.set_title("Étape 4: Trajectoire Finale de Couverture", fontsize=16)
        plt.legend()
        plt.savefig(r"D:\Fructueux\Work\Memoire\Drone Coverage Path Planning\Code\results\stc\zone_3_step_4.png")
        plt.show(block=False)
        plt.pause(0.1)
        
    def wait_for_close(self):
        """Attend que l'utilisateur ferme la fenêtre."""
        print("\nVisualisation terminée. Fermez la fenêtre pour continuer.")
        plt.show()