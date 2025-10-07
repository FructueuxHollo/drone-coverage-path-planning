import numpy as np
import shapely
import shapely.ops
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
import random

class CellularBoustrophedonPlanner:
    """
    Planificateur Boustrophédon utilisant une décomposition cellulaire explicite
    pour gérer les obstacles complexes et fournir une visualisation détaillée.
    """
    def __init__(self, zone_poly, altitude, fov, obstacles_poly, start_coords, overlap):
        self.altitude = altitude
        self.fov = fov
        self.overlap = overlap / 100.0
        self.start_coords = list(start_coords) if start_coords else None

        boundary = shapely.Polygon(zone_poly)
        self.mission_area = boundary
        if obstacles_poly:
            obstacles = shapely.MultiPolygon([shapely.Polygon(p) for p in obstacles_poly])
            self.mission_area = boundary.difference(obstacles)

        self.sweep_offset = self._get_sweep_offset()
        self.cells = []
        self.path_order = []

    def _get_sweep_offset(self):
        """Calcule la distance entre deux lignes de balayage."""
        fov_rad = self.fov * math.pi / 180.0
        return abs(2 * self.altitude * math.tan(fov_rad / 2) * (1 - self.overlap))

    def _decompose_into_cells(self):
        """Découpe la zone de mission en polygones monotones (cellules)."""
        print("Étape 1: Décomposition de la zone en cellules...")
        if self.mission_area.is_empty:
            self.cells = []
            return
        
        bounds = self.mission_area.bounds
        min_x, min_y, max_x, max_y = bounds
        
        # Trouver les coordonnées Y critiques (tous les sommets)
        critical_y = set()
        polygons = [self.mission_area] if isinstance(self.mission_area, shapely.Polygon) else list(self.mission_area.geoms)
        
        for poly in polygons:
            critical_y.update(y for x, y in poly.exterior.coords)
            for interior in poly.interiors:
                critical_y.update(y for x, y in interior.coords)
        
        # Créer des lignes de découpe horizontales individuelles
        individual_slicers = [shapely.LineString([(min_x, y), (max_x, y)]) for y in sorted(list(critical_y))]
        
        # Commencer avec la ou les zones de mission initiales
        geometries_to_split = list(self.mission_area.geoms) if hasattr(self.mission_area, 'geoms') else [self.mission_area]
        
        # Appliquer chaque ligne de découpe l'une après l'autre
        for slicer in individual_slicers:
            newly_split_geometries = []
            for geom in geometries_to_split:
                # La fonction split retourne une GeometryCollection
                split_result = shapely.ops.split(geom, slicer)
                newly_split_geometries.extend(list(split_result.geoms))
            geometries_to_split = newly_split_geometries

        # Filtrer le résultat final pour ne garder que les polygones valides
        self.cells = [p for p in geometries_to_split if isinstance(p, shapely.Polygon) and p.is_valid and not p.is_empty]
        print(f"-> Zone décomposée en {len(self.cells)} cellules.")

    def _plan_path_for_cell(self, cell):
        """Génère un chemin de balayage pour une seule cellule simple."""
        cell_waypoints = []
        bounds = cell.bounds
        min_x, min_y, max_x, max_y = bounds
        
        sweep_x = min_x
        going_up = True

        # Ajustement pour s'assurer que le balayage commence à l'intérieur de la cellule
        if not cell.covers(shapely.Point(sweep_x, min_y)):
             sweep_x += self.sweep_offset / 2

        while sweep_x <= max_x:
            sweep_line = shapely.LineString([(sweep_x, min_y-1), (sweep_x, max_y+1)])
            intersection = cell.intersection(sweep_line)
            
            if not intersection.is_empty and isinstance(intersection, (shapely.LineString, shapely.MultiLineString)):
                if hasattr(intersection, 'geoms'): # MultiLineString
                    # Prendre le segment le plus long en cas de géométrie étrange
                    lines = sorted(list(intersection.geoms), key=lambda line: line.length)
                    coords = list(lines[-1].coords)
                else: # LineString
                    coords = list(intersection.coords)

                start, end = coords[0], coords[-1]
                if not going_up:
                    start, end = end, start
                cell_waypoints.append(start)
                cell_waypoints.append(end)

            going_up = not going_up
            sweep_x += self.sweep_offset
            
        return [list(p) for p in cell_waypoints]
    
    def _densify_path(self, waypoints, spacing):
        """
        Ajoute des points intermédiaires à une trajectoire.
        'waypoints': La liste de points de passage [ [x1,y1], [x2,y2], ... ].
        'spacing': La distance maximale souhaitée entre deux points.
        """
        if spacing <= 0:
            return waypoints
            
        densified_path = []
        if not waypoints:
            return densified_path
            
        densified_path.append(waypoints[0]) # Ajouter le premier point
        
        for i in range(len(waypoints) - 1):
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i+1])
            
            dist = np.linalg.norm(p2 - p1)
            
            if dist > spacing:
                num_segments = int(math.ceil(dist / spacing))
                # Utiliser np.linspace pour générer les points intermédiaires
                intermediate_points = np.linspace(p1, p2, num_segments + 1)
                # Ajouter les nouveaux points (sauf le premier, qui est déjà p1)
                densified_path.extend([list(p) for p in intermediate_points[1:]])
            else:
                densified_path.append(list(p2))
                
        return densified_path

    def plan_coverage_path(self, intermediate_spacing=10.0):
        """Fonction principale orchestrant la planification."""
        if self.mission_area.is_empty: return []

        self._decompose_into_cells()

        print("Étape 2: Planification de la trajectoire pour chaque cellule...")
        cell_paths = {i: self._plan_path_for_cell(cell) for i, cell in enumerate(self.cells)}
        
        # Filtrer les cellules qui n'ont pas généré de chemin
        cell_paths = {i: path for i, path in cell_paths.items() if path}
        if not cell_paths: return []

        print("Étape 3: Connexion des trajectoires cellulaires...")
        
        # Créer un itinéraire global avec un algorithme du plus proche voisin
        remaining_indices = set(cell_paths.keys())

        if not remaining_indices: return []
        
        # Démarrer par la cellule la plus basse et à gauche
        start_cell_idx = min(remaining_indices, key=lambda i: (self.cells[i].bounds[1], self.cells[i].bounds[0]))
        
        current_idx = start_cell_idx
        self.path_order.append(current_idx)
        remaining_indices.remove(current_idx)
        
        # Construire l'ordre du chemin
        while remaining_indices:
            last_point = cell_paths[current_idx][-1]
            
            # Trouver le prochain point d'entrée le plus proche parmi les cellules restantes
            best_next_idx = -1
            min_dist = float('inf')
            
            for next_idx in remaining_indices:
                next_path = cell_paths[next_idx]
                dist_to_start = distance.euclidean(last_point, next_path[0])
                dist_to_end = distance.euclidean(last_point, next_path[-1])
                
                if min(dist_to_start, dist_to_end) < min_dist:
                    min_dist = min(dist_to_start, dist_to_end)
                    best_next_idx = next_idx
            
            current_idx = best_next_idx
            self.path_order.append(current_idx)
            remaining_indices.remove(current_idx)
        
        full_path = []
        for i, idx in enumerate(self.path_order):
            path = cell_paths[idx]
            if i > 0:
                last_point = full_path[-1]
                if distance.euclidean(last_point, path[-1]) < distance.euclidean(last_point, path[0]):
                    path.reverse()
            full_path.extend(path)
            
        if not full_path: return []

        print(f"-> Trajectoire complète de {len(full_path)} points générée.")
        
        # --- ÉTAPE 4: DENSIFICATION DE LA TRAJECTOIRE ---
        print(f"Étape 4: Densification de la trajectoire (espacement de {intermediate_spacing}m)...")
        densified_path = self._densify_path(full_path, spacing=intermediate_spacing)
        print(f"-> Trajectoire densifiée avec {len(densified_path)} points.")

        # --- ÉTAPE 5: Intégration du point de départ ---
        if self.start_coords:
            print("Étape 5: Intégration du point de départ...")
            wp_array = np.array(densified_path)
            closest_idx = distance.cdist([self.start_coords], wp_array).argmin()
            reordered_path = np.roll(wp_array, -closest_idx, axis=0).tolist()
            return [self.start_coords] + reordered_path
            
        return densified_path

# --- INTERFACE MATLAB ---
def planifier_mission(zone_poly, altitude, fov, obstacles_poly, start_coords=None, overlap=0):
    """
    Fonction d'interface pour MATLAB qui utilise NumPy pour convertir de manière fiable
    les types de données MATLAB en listes Python natives.
    """
    print("--- Appel Python (Cellular Boustrophedon) depuis MATLAB ---")
    
    # --- Fonction utilitaire de conversion ---
    def _convert_matlab_data(data):
        # La conversion clé : MATLAB data -> NumPy Array -> Python List
        return np.array(data, dtype=float).tolist()

    try:
        # --- CONVERSION DES DONNÉES VIA NUMPY ---
        zone_poly_native = _convert_matlab_data(zone_poly)
        
        obstacles_poly_native = []
        if obstacles_poly:
            # Gérer le cas où il n'y a qu'un seul obstacle
            if not hasattr(obstacles_poly, '__len__') or len(obstacles_poly) == 0 or not hasattr(obstacles_poly[0], '__len__'):
                 obstacles_poly = [obstacles_poly]
            # Itérer sur la liste d'obstacles et les convertir
            for poly in obstacles_poly:
                obstacles_poly_native.append(_convert_matlab_data(poly))

        start_coords_native = _convert_matlab_data(start_coords) if start_coords else None
        
        # --- FIN DE LA CONVERSION ---
        
        # Le reste du code est maintenant garanti de fonctionner avec des données propres
        planner = CellularBoustrophedonPlanner(
            zone_poly=zone_poly_native,
            altitude=altitude,
            fov=fov,
            obstacles_poly=obstacles_poly_native,
            start_coords=start_coords_native,
            overlap=overlap
        )
        
        waypoints = planner.plan_coverage_path()
        
        if waypoints:
            print(f"-> Trajectoire calculée, {len(waypoints)} waypoints retournés à MATLAB.")
            mission_data = {
                'waypoints': waypoints,
                'resolution_m': planner.sweep_offset
            }
            return mission_data
        else:
            print("-> Échec de la planification.")
            return {} # Retourner un dict vide pour la robustesse côté MATLAB

    except Exception as e:
        import traceback
        print(f"Erreur dans le planificateur: {e}")
        traceback.print_exc()
        return {}

# --- SECTION DE TEST STANDALONE ---
if __name__ == '__main__':
    print("--- Lancement en mode standalone avec visualisation détaillée ---")
    
    # Paramètres de la mission
    zone = [
        (0, 0),
        (1200, 0),
        (1200, 900),
        (0, 900)
    ]

    # Cette zone n'a pas d'obstacles internes dans l'image.
    obstacles = [
        # Obstacle I
        [(200, 500), (450, 500), (450, 700), (200, 700)],

        # Obstacle II
        [(150, 100), (550, 100), (550, 300), (150, 300)],

        # Obstacle III
        [(750, 600), (900, 600), (900, 800), (750, 800)],

        # Obstacle IV
        [(800, 200), (1000, 200), (1000, 350), (800, 350)],
    ]
    start_pt = [50.0, 70.0]
    
    # 1. Instancier et exécuter le planificateur
    planner = CellularBoustrophedonPlanner(zone, 43, 60, obstacles, start_pt, 0)
    final_waypoints = planner.plan_coverage_path()

    # 2. Visualisation
    if final_waypoints:
        # Création de la figure avec deux sous-graphes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle("Analyse de la Planification Boustrophédon", fontsize=16)

        # --- PANNEAU 1: Décomposition Cellulaire ---
        ax1.set_title("1. Décomposition en Cellules (Ordre de Visite)")
        ax1.set_aspect('equal', adjustable='box')
        ax1.plot(*np.array(zone).T, 'b-', label='Zone de mission')
        for obs in obstacles:
            ax1.fill(*np.array(obs).T, 'r', alpha=0.5, label='Obstacle')
        
        # Dessiner et numéroter chaque cellule
        for i, cell_idx in enumerate(planner.path_order):
            cell = planner.cells[cell_idx]
            color = plt.cm.viridis(i / len(planner.path_order)) # Dégradé de couleur
            ax1.fill(*cell.exterior.xy, color=color, alpha=0.6)
            # Ajouter le numéro de visite au centre de la cellule
            ax1.text(cell.centroid.x, cell.centroid.y, str(i + 1), 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.grid(True)
        ax1.legend()

        # --- PANNEAU 2: Trajectoire Finale ---
        ax2.set_title("2. Trajectoire Finale Connectée avec Direction")
        ax2.set_aspect('equal', adjustable='box')
        ax2.plot(*np.array(zone).T, 'b-', label='Zone de mission')
        for obs in obstacles:
            ax2.fill(*np.array(obs).T, 'r', alpha=0.5, label='Obstacle')

        wp_array = np.array(final_waypoints)
        ax2.plot(wp_array[:, 0], wp_array[:, 1], 'k-', linewidth=2.0, alpha=0.8, label='Trajectoire')
        ax2.plot(start_pt[0], start_pt[1], 'm*', markersize=15, label='Point de Départ')

        # Ajouter des flèches directionnelles
        for i in range(0, len(wp_array) - 1, 5): # Mettre une flèche tous les 5 segments
            p1 = wp_array[i]
            p2 = wp_array[i+1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            if math.hypot(dx, dy) > 0.1: # Éviter les flèches nulles
                 ax2.arrow(p1[0], p1[1], dx, dy, head_width=8, head_length=8, fc='green', ec='green', length_includes_head=True)

        ax2.set_xlabel('East (m)')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(r"D:\Fructueux\Work\Memoire\Drone Coverage Path Planning\Code\results\boustrophedon\zone_3.png")
        plt.show()