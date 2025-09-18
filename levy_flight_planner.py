import numpy as np
import shapely
import shapely.ops
import math

class LevyFlightPlanner:
    """
    Génère une trajectoire de couverture imprévisible basée sur une marche
    aléatoire de type "Lévy Flight", contrainte à l'intérieur d'une zone de mission.
    """
    def __init__(self, zone_poly, obstacles_poly, start_coords, altitude, fov, overlap):
        # --- 1. Préparation de la zone de mission ---
        self.offset = self._calculate_offset(altitude, fov, overlap)
        print(f"Marge de sécurité (offset) calculée : {self.offset:.2f} m")
        boundary = shapely.Polygon(zone_poly)
        safe_area = boundary
        if obstacles_poly:
            obstacles = shapely.MultiPolygon([shapely.Polygon(p) for p in obstacles_poly])
            safe_area = boundary.difference(obstacles)

        # "Réduction" de la zone de mission pour créer une marge de sécurité
        self.safe_area = safe_area.buffer(-3)

        # S'assurer que le point de départ est bien dans la zone
        start_point_geom = shapely.Point(start_coords)
        if not self.safe_area.contains(start_point_geom):
            # Si le point est dehors, on trouve le point valide le plus proche sur la frontière de la zone de sécurité 
            nearest_valid_point = shapely.ops.nearest_points(self.safe_area.boundary, start_point_geom)[0]
            self.start_coords = [nearest_valid_point.x, nearest_valid_point.y]
        else:
            self.start_coords = list(start_coords)
            
        print("Planificateur Lévy Flight initialisé.")
    
    def _calculate_offset(self, altitude, fov, overlap):
        """Calcule la distance de sécurité (identique à la résolution Boustrophédon)."""
        overlap_ratio = overlap / 100.0
        fov_rad = fov * math.pi / 180.0
        return abs(2 * altitude * math.tan(fov_rad / 2) * (1 - overlap_ratio))

    def _generate_levy_step(self, alpha=1.2, scale=10.0):
        """
        Génère une longueur de pas à partir d'une distribution de Lévy.
        - alpha: Paramètre de stabilité (entre 1 et 2). Plus il est bas, plus les sauts longs sont fréquents.
        - scale: Facteur d'échelle pour la taille des pas (en mètres).
        """
        # Algorithme de Mantegna pour générer des variables de Lévy
        u = np.random.normal(0, 1) * np.sqrt(math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / (math.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))
        v = np.random.normal(0, 1)
        
        step = u / (abs(v)**(1 / alpha))
        
        return scale * step
    
    def _densify_path(self, waypoints, spacing):
        """Ajoute des points intermédiaires à une trajectoire."""
        if spacing <= 0: return waypoints
        densified_path = []
        if not waypoints: return densified_path
        
        densified_path.append(waypoints[0])
        for i in range(len(waypoints) - 1):
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i+1])
            dist = np.linalg.norm(p2 - p1)
            if dist > spacing:
                num_segments = int(math.ceil(dist / spacing))
                intermediate_points = np.linspace(p1, p2, num_segments + 1)
                densified_path.extend([list(p) for p in intermediate_points[1:]])
            else:
                densified_path.append(list(p2))
        return densified_path

    def plan_coverage_path(self, total_path_length=2000.0, max_step_factor=0.5, intermediate_spacing=10.0):
        """
        Construit la trajectoire de vol de Lévy.
        - total_path_length: Longueur totale de la trajectoire à générer (en mètres).
        - max_step_factor: Limite la taille maximale d'un saut à un pourcentage de la diagonale de la zone.
        """
        if self.safe_area.is_empty:
            print("Erreur: La zone de sécurité est vide après application de l'offset. L'offset est peut-être trop grand.")
            return []

        waypoints = [self.start_coords]
        current_length = 0.0
        
        bounds = self.safe_area.bounds
        max_dim = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
        max_step_length = max_dim * max_step_factor

        print(f"Étape 1: Lancement de la génération du chemin (objectif: {total_path_length}m)...")
        
        while current_length < total_path_length:
            current_point = waypoints[-1]
            
            # Tenter de trouver un pas valide (qui reste dans la zone)
            for _ in range(100): # Maximum 100 tentatives pour éviter une boucle infinie
                direction = np.random.uniform(0, 2 * np.pi)
                step_length = self._generate_levy_step()
                
                # Brider la longueur du pas pour éviter les sauts absurdes
                step_length = min(abs(step_length), max_step_length)

                # Calculer le prochain point potentiel
                next_point = [
                    current_point[0] + step_length * np.cos(direction),
                    current_point[1] + step_length * np.sin(direction)
                ]
                
                # --- Vérification de la validité du segment ---
                line_segment = shapely.LineString([current_point, next_point])
                
                if self.safe_area.contains(line_segment):
                    # Le segment est valide, on l'ajoute
                    waypoints.append(next_point)
                    current_length += step_length
                    break # Sortir de la boucle de tentatives
            else:
                # Si après 100 tentatives aucun pas valide n'est trouvé, on arrête
                print("Avertissement: Impossible de trouver un pas valide. Arrêt prématuré.")
                break
        
        print(f"-> Trajectoire de Lévy brute générée avec {len(waypoints)} points.")
        print(f"Étape 2: Densification de la trajectoire (espacement de {intermediate_spacing}m)...")
        densified_waypoints = self._densify_path(waypoints, spacing=intermediate_spacing)
        print(f"-> Trajectoire finale densifiée avec {len(densified_waypoints)} points.")
        
        return densified_waypoints
        return waypoints


# --- INTERFACE MATLAB ---
def planifier_mission(zone_poly, altitude, fov, obstacles_poly, start_coords=None, overlap=0):
    """
    Fonction d'interface pour MATLAB qui utilise NumPy pour la conversion des données.
    """
    print("--- Appel Python (Lévy Flight Planner) depuis MATLAB ---")
    
    def _convert_matlab_data(data):
        return np.array(data, dtype=float).tolist()
    
    try:
        zone_poly_native = _convert_matlab_data(zone_poly)
        obstacles_poly_native = []
        if obstacles_poly:
            if not hasattr(obstacles_poly, '__len__') or not hasattr(obstacles_poly[0], '__len__'):
                 obstacles_poly = [obstacles_poly]
            for poly in obstacles_poly:
                obstacles_poly_native.append(_convert_matlab_data(poly))
        start_coords_native = _convert_matlab_data(start_coords) if start_coords else None
        
        planner = LevyFlightPlanner(
            zone_poly=zone_poly_native,
            obstacles_poly=obstacles_poly_native,
            start_coords=start_coords_native,
            altitude=altitude,
            fov=fov,
            overlap=overlap
        )
        
        # Vous pouvez ajuster la longueur totale de la mission ici
        waypoints = planner.plan_coverage_path(total_path_length=10000.0)
        
        if waypoints:
            mission_data = {
                'waypoints': waypoints,
                'resolution_m': planner.offset 
            }
            return mission_data
        else:
            print("-> Échec de la planification.")
            return {}

    except Exception as e:
        import traceback
        print(f"Erreur dans le planificateur: {e}")
        traceback.print_exc()
        return {}

# --- Section pour le test standalone ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("--- Lancement du planificateur Lévy Flight en mode standalone ---")
    
    # Définition de la mission pour le test
    zone = [
        [50, 50], [450, 50], [450, 350], [300, 350], [300, 150],
        [200, 150], [200, 350], [50, 350], [50, 50]
    ]
    
    obstacles = [
        [[100, 100], [150, 100], [150, 150], [100, 150]]
    ]
    
    # Un point de départ bien au centre de la zone libre
    start_pt = [125.0, 125.0]
    
    # 1. Créer une instance du planificateur
    planner = LevyFlightPlanner(
        zone_poly=zone,
        obstacles_poly=obstacles,
        start_coords=start_pt,
        altitude=40,
        fov=60,
        overlap=20
    )
    
    # 2. Planifier la trajectoire (ajuster la longueur pour la visualisation)
    final_waypoints = planner.plan_coverage_path(total_path_length=50000.0)
    
    # 3. Si la planification a réussi, générer la visualisation
    if final_waypoints:
        print(f"\nPlanification réussie, {len(final_waypoints)} waypoints générés.")
        
        # Conversion en array NumPy pour un affichage facile
        wp_array = np.array(final_waypoints)
        zone_array = np.array(zone)
        
        plt.figure(figsize=(10, 8))
        
        # Dessiner la zone de mission
        plt.plot(zone_array[:, 0], zone_array[:, 1], 'b-', label='Zone de mission')
        
        # Dessiner les obstacles
        for obs_coords in obstacles:
            obs_array = np.array(obs_coords)
            plt.fill(obs_array[:, 0], obs_array[:, 1], 'r', alpha=0.5, label='Obstacle')
        
        # Dessiner la trajectoire Lévy Flight
        # 'k-o' -> Ligne (k-) noire avec des petits cercles (o) à chaque waypoint
        plt.plot(wp_array[:, 0], wp_array[:, 1], 'k-o', 
                 markersize=2, linewidth=1, label='Trajectoire Lévy Flight')
        
        # Marquer le point de départ
        plt.plot(start_pt[0], start_pt[1], 'm*', markersize=15, label='Point de Départ')

        plt.title('Visualisation de la Trajectoire Lévy Flight')
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.axis('equal')
        plt.grid(True)
        
        # Gérer les légendes pour éviter les doublons
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.show()
    else:
        print("\nÉchec de la planification.")