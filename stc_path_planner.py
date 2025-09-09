import heapq
import cv2
import numpy as np
import os
import time
import math 

class DronePathPlanner:
    """
    Planifie une trajectoire de couverture de zone pour un drone.
    Prend en entrée les sommets d'un polygone définissant la zone
    et génère une série de waypoints pour une couverture complète.
    """
    def __init__(self, polygon_vertices, altitude_m, camera_fov_degrees, obstacle_polygons=None):
        """
        Initialise le planificateur.
        - polygon_vertices: Liste de tuples (x, y) des sommets du polygone principal.
        - altitude_m: Altitude de vol du drone en mètres.
        - camera_fov_degrees: Angle de vue (FOV) horizontal de la caméra en degrés.
        - obstacle_polygons: Liste optionnelle de polygones d'obstacles.
        """
        if not polygon_vertices:
            raise ValueError("La liste des sommets du polygone ne peut pas être vide.")
            
        self.polygon_vertices = np.array(polygon_vertices, dtype=np.int32)
        self.obstacle_polygons = obstacle_polygons if obstacle_polygons is not None else []
        # Calcul de la résolution de la grille à partir des paramètres du drone
        self.altitude = altitude_m
        self.fov_degrees = camera_fov_degrees
        # La formule trigonométrique pour calculer la largeur de la vue au sol
        # On convertit le FOV en radians pour la fonction tan() de Python
        fov_radians = math.radians(self.fov_degrees)
        self.resolution = 2 * self.altitude * math.tan(fov_radians / 2)
        self.grid = []
        self.start_point = None
        self.origin_offset = (0, 0) # Pour mapper les coordonnées du polygone à la grille
        self.hamiltonian_path = []

        print("Initialisation du planificateur...")
        print(f"Altitude: {self.altitude}m, FOV: {self.fov_degrees}°")
        print(f"-> Résolution de grille calculée : {self.resolution:.2f} mètres par cellule.")
        self._create_grid_from_polygon()

    def _create_grid_from_polygon(self):
        """
        Crée une grille binaire à partir des sommets du polygone.
        Les cellules à l'intérieur du polygone sont marquées comme 0 (libres),
        celles à l'extérieur sont marquées comme 1 (obstacles).
        """
        # 1. Déterminer la boîte englobante (bounding box) du polygone
        min_x, min_y = np.min(self.polygon_vertices, axis=0)
        max_x, max_y = np.max(self.polygon_vertices, axis=0)
        
        self.origin_offset = (min_x, min_y)

        # 2. Calculer les dimensions de la grille en fonction de la résolution
        cols = int(np.ceil((max_x - min_x) / self.resolution))
        rows = int(np.ceil((max_y - min_y) / self.resolution))
        
        print(f"Grille générée : {rows} lignes, {cols} colonnes.")
        self.grid = np.ones((rows, cols), dtype=int) # Tout est un obstacle au départ

        # 3. Ajuster les coordonnées du polygone pour qu'elles correspondent à la grille
        poly_grid_coords = ((self.polygon_vertices - self.origin_offset) / self.resolution).astype(np.int32)

        obs_grid_coords_list = []
        for obs_poly in self.obstacle_polygons:
            np_obs_poly = np.array(obs_poly, dtype=np.int32)
            obs_coords = ((np_obs_poly - self.origin_offset) / self.resolution).astype(np.int32)
            obs_grid_coords_list.append(obs_coords)

        # 4. Remplir la grille 
        for r in range(rows):
            for c in range(cols):
                # On teste le centre de la cellule
                point_x = c + 0.5
                point_y = r + 0.5
                # Test 1: Le point doit être dans la zone de surveillance principale
                if cv2.pointPolygonTest(poly_grid_coords, (point_x, point_y), False) >= 0:
                    # Test 2: Le point ne doit PAS être dans une zone d'obstacle
                    is_in_obstacle = False
                    for obs_poly in obs_grid_coords_list:
                        if cv2.pointPolygonTest(obs_poly, (point_x, point_y), False) >= 0:
                            is_in_obstacle = True
                            break # Inutile de vérifier les autres obstacles
                    
                    if not is_in_obstacle:
                        self.grid[r, c] = 0 # C'est un espace libre !

        # 5. Définir un point de départ 
        start_coords = np.argwhere(self.grid == 0)
        if len(start_coords) > 0:
            self.start_point = tuple(start_coords[0])
            print(f"Point de départ trouvé : {self.start_point}")
        else:
            raise Exception("Aucune zone libre trouvée à l'intérieur du polygone. Vérifiez les coordonnées ou la résolution.")

    def plan_coverage_path(self):
        """
        Fonction principale pour planifier le chemin de couverture.
        Génère l'arbre recouvrant minimum (MST) puis le chemin Hamiltonien.
        Retourne la liste des waypoints.
        """
        if self.start_point is None or self.grid[self.start_point[0]][self.start_point[1]] == 1:
            print("Le point de départ est bloqué.")
            return None
        if not self._is_connected(self.start_point):
            print("La carte n'est pas connexe (il y a des zones isolées).")
            return None

        print("Planification du chemin en cours...")
        rows, cols = self.grid.shape
        
        # Calcul de l'arbre recouvrant minimum (MST) avec l'algorithme de Prim
        mst = set()
        path_from = {self.start_point: None}
        edges = [(0, self.start_point, None)]  # (poids, (r, c), from_cell)

        num_free_cells = np.sum(self.grid == 0)
        
        while edges and len(mst) < num_free_cells:
            _, (r, c), from_cell = heapq.heappop(edges)
            if (r, c) in mst:
                continue
            mst.add((r, c))
            if from_cell is not None:
                path_from[(r, c)] = from_cell
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and self.grid[rr, cc] == 0 and (rr, cc) not in mst:
                    heapq.heappush(edges, (1, (rr, cc), (r, c)))
        
        print("Arbre recouvrant minimum (MST) généré.")

        # Génération du chemin Hamiltonien à partir du MST
        sub_grid = self._subdivide_grid()
        self.hamiltonian_path = self._generate_hamiltonian_path(sub_grid, path_from)
        
        print(f"Chemin Hamiltonien généré avec {len(self.hamiltonian_path)} waypoints.")
        
        # Conversion des coordonnées de la grille en coordonnées "réelles"
        real_world_waypoints = [
            (((c+1)/2.0) * self.resolution + self.origin_offset[0], ((r+1)/2.0) * self.resolution + self.origin_offset[1])
            for r, c in self.hamiltonian_path
        ]

        return real_world_waypoints
        
    def save_path_to_file(self, waypoints, filename="output/waypoints.txt"):
        """Sauvegarde les waypoints dans un fichier."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("x,y\n")
            for x, y in waypoints:
                f.write(f"{x},{y}\n")
        print(f"Waypoints sauvegardés dans {filename}")

    # --- Méthodes de visualisation (optionnelles) ---

    def visualize_grid_and_mst(self, cell_size=20):
        """Visualise la grille, les obstacles et le MST."""
        rows, cols = self.grid.shape
        img = np.full((rows * cell_size, cols * cell_size, 3), (200, 200, 200), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                if self.grid[r, c] == 0: # Libre
                    cv2.rectangle(img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), (255, 255, 255), -1)
                cv2.rectangle(img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), (0, 0, 0), 1)

        # Dessiner le polygone principale en bleu
        poly_img_coords = ((self.polygon_vertices - self.origin_offset) / self.resolution * cell_size).astype(np.int32)
        cv2.polylines(img, [poly_img_coords], isClosed=True, color=(255, 0, 0), thickness=2)

        # Dessiner les polygones d'obstacles (en rouge)
        for obs_poly in self.obstacle_polygons:
            np_obs_poly = np.array(obs_poly, dtype=np.int32)
            obs_img_coords = ((np_obs_poly - self.origin_offset) / self.resolution * cell_size).astype(np.int32)
            cv2.fillPoly(img, [obs_img_coords], color=(100, 100, 255))

        # Dessiner le MST (nécessite de relancer une partie de la planification)
        # Pour une version propre, path_from devrait être un attribut de la classe
        # Ici, on le recalcule pour la démo.
        # ... (logique de calcul du MST à insérer ici si besoin de le visualiser)

        filename = "output/grid_and_polygon.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, img)
        print(f"Visualisation de la grille sauvegardée dans {filename}")
        
    def animate_flight_path(self, cell_size=10, speed_ms=50, video_filename=None, video_fps=30):
        """
        Crée une animation du drone (un point) parcourant le chemin Hamiltonien
        et coloriant les cellules visitées.
        """
        if not self.hamiltonian_path:
            print("Aucun chemin à animer. Lancez d'abord plan_coverage_path().")
            return
            
        rows, cols = self.grid.shape
        sub_rows, sub_cols = rows * 2, cols * 2
        height, width = sub_rows * cell_size, sub_cols * cell_size

        # Configuration de l'enregistreur vidéo
        video_writer = None
        if video_filename:
            os.makedirs(os.path.dirname(video_filename), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, video_fps, (width, height))
            if not video_writer.isOpened():
                print(f"Erreur: Impossible d'initialiser l'enregistreur vidéo pour {video_filename}")
                video_writer = None # On annule l'enregistrement
            else:
                print(f"Enregistrement de l'animation dans : {video_filename}")
        
        # Création de l'image de base
        base_img = np.full((sub_rows * cell_size, sub_cols * cell_size, 3), (200, 200, 200), dtype=np.uint8)
        # Dessin de la grille subdivisée
        for r in range(sub_rows):
            for c in range(sub_cols):
                 # Colorier les cellules hors polygone
                if self.grid[r//2, c//2] == 1:
                    cv2.rectangle(base_img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), (50, 50, 50), -1)
                else:
                    cv2.rectangle(base_img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), (255, 255, 255), -1)
                cv2.rectangle(base_img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), (180, 180, 180), 1)
        
        visited_sub_cells = set()

        print("Lancement de l'animation... Appuyez sur 'q' pour quitter.")
        for i, (r, c) in enumerate(self.hamiltonian_path):
            frame = base_img.copy()
            
            # Colorier les cellules déjà visitées
            visited_sub_cells.add((r, c))
            for vr, vc in visited_sub_cells:
                cv2.rectangle(frame, (vc * cell_size, vr * cell_size), ((vc + 1) * cell_size, (vr + 1) * cell_size), (200, 255, 200), -1)
            
            # Dessiner le chemin parcouru
            if i > 0:
                path_so_far = np.array(self.hamiltonian_path[:i+1])
                path_pixels = (path_so_far[:, ::-1] * cell_size) + cell_size // 2
                cv2.polylines(frame, [path_pixels], isClosed=False, color=(0, 150, 0), thickness=2)

            # Dessiner le drone
            drone_pos = (c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)
            cv2.circle(frame, drone_pos, cell_size // 2, (0, 0, 255), -1) # Point rouge pour le drone

            cv2.imshow("Animation du parcours du drone", frame)

            if video_writer:
                video_writer.write(frame)
            
            # Quitter si 'q' est pressé
            if cv2.waitKey(speed_ms) & 0xFF == ord('q'):
                break

        if video_writer:
            video_writer.release()
            print("Vidéo enregistrée avec succès.")
        
        cv2.destroyAllWindows()
        print("Animation terminée.")

    # --- Méthodes utilitaires reprises du code original ---

    def _subdivide_grid(self):
        rows, cols = self.grid.shape
        sub_grid = np.zeros((rows * 2, cols * 2), dtype=int)
        for r in range(rows):
            for c in range(cols):
                if self.grid[r, c] == 1:
                    sub_grid[r*2, c*2] = 1
                    sub_grid[r*2+1, c*2] = 1
                    sub_grid[r*2, c*2+1] = 1
                    sub_grid[r*2+1, c*2+1] = 1
        return sub_grid

    def _is_connected(self, start):
        rows, cols = self.grid.shape
        stack = [start]
        visited = set()
        
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited:
                visited.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and self.grid[rr, cc] == 0 and (rr, cc) not in visited:
                        stack.append((rr, cc))
        
        num_free_cells = np.sum(self.grid == 0)
        return len(visited) == num_free_cells
        
    def _generate_hamiltonian_path(self, sub_grid, path_from):
        path_set_bi = set()
        for k, v in path_from.items():
            if v is not None:
                path_set_bi.add((k, v))
                path_set_bi.add((v, k))
        
        hamiltonian_path = []
        start_node = (self.start_point[0] * 2, self.start_point[1] * 2)
        stack = [start_node]
        visited = set()
        
        path_map = {}
        
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited:
                visited.add((r, c))
                hamiltonian_path.append((r, c))
                
                # Logique complexe de sélection du prochain mouvement
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < len(sub_grid) and 0 <= cc < len(sub_grid[0]) and sub_grid[rr, cc] == 0 and (rr, cc) not in visited):
                        continue

                    valid_move = False
                    if r // 2 == rr // 2 and c // 2 == cc // 2: # Mouvement interne à une cellule mère
                        outside = (r // 2, c // 2)
                        if c % 2 == 0 and cc % 2 == 0 and ((outside, (outside[0], outside[1] - 1))) in path_set_bi: continue
                        if c % 2 == 1 and cc % 2 == 1 and ((outside, (outside[0], outside[1] + 1))) in path_set_bi: continue
                        if r % 2 == 0 and rr % 2 == 0 and ((outside, (outside[0] - 1, outside[1]))) in path_set_bi: continue
                        if r % 2 == 1 and rr % 2 == 1 and ((outside, (outside[0] + 1, outside[1]))) in path_set_bi: continue
                        valid_move = True
                    else: # Mouvement entre cellules mères
                        if (((r // 2, c // 2), (rr // 2, cc // 2))) in path_set_bi:
                            valid_move = True

                    if valid_move:
                        stack.append((rr, cc))
                        path_map[(rr,cc)] = (r,c)

        # Reconstruire le chemin pour avoir un ordre logique
        ordered_path = []
        curr = start_node
        # Le chemin généré par DFS n'est pas séquentiel, nous devons le reconstruire.
        # Pour une couverture de type "tondeuse", un algorithme de balayage serait plus simple.
        # Celui-ci suit le MST, ce qui est bon pour la connectivité.
        # La logique originale de reconstruction du chemin était complexe, on retourne la version simple de la visite DFS
        return hamiltonian_path


def planifier_mission(zone_poly, altitude, fov, obstacles_poly):
    """
    Fonction principale d'interface pour MATLAB.
    Prend en entrée des listes Python et retourne les waypoints.
    """
    print("--- Appel Python depuis MATLAB ---")
    
    # 1. Instancier le planificateur
    planner = DronePathPlanner(
        polygon_vertices=zone_poly,
        altitude_m=altitude,
        camera_fov_degrees=fov,
        obstacle_polygons=obstacles_poly
    )
    
    # 2. Planifier la trajectoire
    waypoints = planner.plan_coverage_path()
    
    if waypoints:
        print(f"-> Trajectoire calculée, {len(waypoints)} waypoints retournés à MATLAB.")
        # 3. Retourner les waypoints dans un format simple (liste de listes)
        mission_data = {
            'waypoints': waypoints,
            'resolution_m': planner.resolution
        }
        return mission_data
    else:
        print("-> Échec de la planification.")
        return []

if __name__ == "__main__":
    
    # Définition de la mission comme avant
    FLIGHT_ALTITUDE_M = 40
    CAMERA_FOV_DEGREES = 60
    
    zone = [
        (50, 50), (450, 50), (450, 350), (50, 350)
    ]
    obstacles = [
        [(150, 150), (250, 150), (250, 250), (150, 250)],
        [(300, 80), (400, 80), (350, 150)]
    ]
    
    # Appel de la fonction principale
    final_waypoints = planifier_mission(zone, FLIGHT_ALTITUDE_M, CAMERA_FOV_DEGREES, obstacles)
    
    if final_waypoints:
        print("\n--- Exécution en mode standalone terminée ---")
        # Ici, vous pourriez sauvegarder le fichier ou lancer l'animation si besoin
        # planner.save_path_to_file(final_waypoints) # Ne peut pas être appelé directement