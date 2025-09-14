import shapely.geometry as sg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class STCVisualizer:
    """
    Classe dédiée à la visualisation des différentes étapes de l'algorithme STC.
    """
    def __init__(self, work_area, obstacles):
        self.work_area = work_area
        self.obstacles = obstacles
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Spanning Tree Coverage Planner")
        
        # Dessiner l'environnement de base
        self.ax.plot(*work_area.exterior.xy, color='blue', label='Zone de travail')
        for i, obs in enumerate(obstacles):
            label = 'Obstacle' if i == 0 else ''
            self.ax.fill(*obs.exterior.xy, alpha=0.5, color='red', ec='black', label=label)
        
    def draw_grid(self, cell_polygons, title="Étape 2: Grille et Cellules Valides"):
        """Dessine les cellules valides de la grille."""
        print(f"Visualisation: {title}")
        for cell in cell_polygons:
            self.ax.plot(*cell.exterior.xy, color='green', alpha=0.4, linewidth=0.8)
        self.ax.set_title(title)
        self.show()

    def draw_graph(self, graph, cell_map, title="Étape 3: Graphe d'Adjacence"):
        """Dessine le graphe d'adjacence complet."""
        print(f"Visualisation: {title}")
        for edge in graph.edges():
            p1 = cell_map[edge[0]]
            p2 = cell_map[edge[1]]
            self.ax.plot([p1.x, p2.x], [p1.y, p2.y], color='gray', linestyle='--', linewidth=0.7)
        self.ax.set_title(title)
        self.show()

    def draw_spanning_tree(self, tree, cell_map, title="Étape 4: Arbre Couvrant (Spanning Tree)"):
        """Dessine l'arbre couvrant."""
        print(f"Visualisation: {title}")
        for edge in tree.edges():
            p1 = cell_map[edge[0]]
            p2 = cell_map[edge[1]]
            self.ax.plot([p1.x, p2.x], [p1.y, p2.y], color='purple', linestyle='-', linewidth=2.5)
        self.ax.set_title(title)
        self.show()
    
    def animate_path(self, path_polygons, title="Étape 5: Animation du Chemin de Couverture"):
        """Anime le parcours du drone et colore la zone couverte."""
        print(f"Visualisation: {title}")
        self.ax.set_title(title)

        # Créer une collection de patches pour les sous-cellules
        patches = [plt.Rectangle((poly.bounds[0], poly.bounds[1]), poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1],
                                 facecolor='orange', alpha=0.0) for poly in path_polygons]
        for p in patches:
            self.ax.add_patch(p)
        
        # Le drone est représenté par un point
        drone_marker, = self.ax.plot([], [], 'bo', markersize=10, label='Drone')

        def init():
            drone_marker.set_data([], [])
            for p in patches:
                p.set_alpha(0.0)
            return [drone_marker] + patches

        def animate(i):
            # Mettre à jour la position du drone
            current_poly = path_polygons[i]
            center = current_poly.centroid
            drone_marker.set_data([center.x], [center.y])
            
            # Colorer la cellule visitée
            patches[i].set_alpha(0.6)
            
            return [drone_marker] + patches

        # Créer l'animation
        anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                       frames=len(path_polygons), interval=50, 
                                       blit=True, repeat=False)
        # Sauvegarder l'animation si nécessaire
        anim.save('coverage_path.mp4', writer='ffmpeg', fps=10)
        self.show()
        return anim # Retourner l'objet animation pour le garder en mémoire

    def show(self):
        """Affiche la figure."""
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show(block=False) # Non-blocking show
        plt.pause(0.1) # Pause to allow plot to render

class STCPlanner:
    """
    Implémente l'algorithme STC via la méthode de sculpture de graphe.
    """
    def __init__(self, work_area, obstacles, tool_size):
        self.work_area = work_area
        self.obstacles = obstacles
        self.D = tool_size
        self.cell_size = 2 * self.D
        
        # Ces attributs seront peuplés par les différentes méthodes
        self.grid_cells = {} # Liste des indices (i,j) des cellules valides
        self.cell_map = {} # Dico: (i,j) -> centre de la cellule (Point shapely)
        self.graph = None
        self.spanning_tree = None
        self.coverage_path = []
        
    def _create_grid(self):
        """Étape 2: Discrétise la zone et trouve les cellules valides."""
        print("--- Étape 2: Création de la grille grossière ---")
        min_x, min_y, max_x, max_y = self.work_area.bounds
        
        x_coords = np.arange(min_x, max_x, self.cell_size)
        y_coords = np.arange(min_y, max_y, self.cell_size)

        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                cell = sg.box(x, y, x + self.cell_size, y + self.cell_size)
                cell_center = cell.centroid

                is_in_work_area = self.work_area.contains(cell_center)
                has_no_obstacle = not any(cell.intersects(obs) for obs in self.obstacles)

                if is_in_work_area and has_no_obstacle:
                    self.grid_cells.append((i, j))
                    self.cell_polygons[(i, j)] = cell
                    self.cell_map[(i, j)] = cell_center
        
        print(f"{len(self.grid_cells)} cellules valides trouvées.")
        
    def _find_start_cell(self, start_point):
        """Trouve l'indice de la cellule contenant le point de départ."""
        for idx, polygon in self.cell_polygons.items():
            if polygon.contains(start_point):
                return idx
        # Si le point est hors grille, on prend la cellule valide la plus proche
        if not self.grid_cells:
            raise ValueError("Aucune cellule valide n'a été trouvée dans la zone de travail.")
        
        min_dist = float('inf')
        closest_cell = None
        for idx in self.grid_cells:
            dist = start_point.distance(self.cell_map[idx])
            if dist < min_dist:
                min_dist = dist
                closest_cell = idx
        print(f"Avertissement: Le point de départ n'est pas dans une cellule valide. Utilisation de la cellule la plus proche : {closest_cell}")
        return closest_cell

    def _build_graph(self):
        """Étape 3: Construit le graphe d'adjacence."""
        print("--- Étape 3: Construction du graphe ---")
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.grid_cells)
        
        cell_set = set(self.grid_cells)
        for i, j in self.grid_cells:
            # Vérifier les voisins (droite, gauche, haut, bas)
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (i + di, j + dj)
                if neighbor in cell_set:
                    self.graph.add_edge((i, j), neighbor)
        
        print(f"Graphe créé avec {self.graph.number_of_nodes()} nœuds et {self.graph.number_of_edges()} arêtes.")

    def _compute_spanning_tree(self, start_node):
        """Étape 4: Calcule un arbre couvrant (DFS)."""
        print("--- Étape 4: Calcul de l'arbre couvrant ---")
        if not self.graph.has_node(start_node):
             raise ValueError(f"Le nœud de départ {start_node} n'existe pas dans le graphe.")
        if not nx.is_connected(self.graph):
            # Gérer le cas où le graphe n'est pas connexe
            # On ne prend que la composante contenant le point de départ
            nodes_in_component = nx.node_connected_component(self.graph, start_node)
            subgraph = self.graph.subgraph(nodes_in_component)
            self.spanning_tree = nx.dfs_tree(subgraph, source=start_node)
        else:
            self.spanning_tree = nx.dfs_tree(self.graph, source=start_node)
            
        print(f"Arbre couvrant calculé avec {self.spanning_tree.number_of_nodes()} nœuds.")
    def _get_subcell(self, cell_idx, sub_cell_num):
        """Retourne le polygone d'une sous-cellule.
        Numérotation:
        +---+---+
        | 1 | 0 |
        +---+---+
        | 2 | 3 |
        +---+---+
        """
        i, j = cell_idx
        # Le centre de la cellule 2D x 2D
        main_cell_center_x = self.cell_map[(i, j)].x
        main_cell_center_y = self.cell_map[(i, j)].y
        
        offsets = {
            0: (self.D / 2, -self.D / 2),   # Haut-droite
            1: (self.D / 2, self.D / 2),  # Haut-gauche
            2: (-self.D / 2, self.D / 2), # Bas-gauche
            3: (-self.D / 2, -self.D / 2)   # Bas-droite
        }
        
        dx, dy = offsets[sub_cell_num]
        sub_cell_center_x = main_cell_center_x + dx
        sub_cell_center_y = main_cell_center_y + dy
        
        return sg.box(
            sub_cell_center_x - self.D / 2,
            sub_cell_center_y - self.D / 2,
            sub_cell_center_x + self.D / 2,
            sub_cell_center_y + self.D / 2,
        )    

    def _generate_coverage_path(self, start_node):
        """
        Étape 5: Génère le chemin par circumnavigation de l'arbre couvrant.
        Suit rigoureusement la logique STC : parcours DFS avec gestion des sous-cellules
        selon la direction de navigation.
        """
        print("--- Étape 5: Génération du chemin ---")
        self.coverage_path = []

        def get_direction(from_cell, to_cell):
            """Détermine la direction de navigation entre deux cellules adjacentes."""
            fi, fj = from_cell
            ti, tj = to_cell

            if ti > fi:
                return 'east'
            elif ti < fi:
                return 'west'
            elif tj > fj:
                return 'north'
            elif tj < fj:
                return 'south'
            else:
                return None

        def get_subcells_for_direction(direction, type, is_forward=True):
            """
            Retourne les sous-cellules à visiter selon la direction.
            Pour suivre le côté droit de l'arête dans le sens antihoraire.

            Numérotation des sous-cellules:
            +---+---+
            | 1 | 0 |
            +---+---+
            | 2 | 3 |
            +---+---+
            """
            if type == "crossing":
                if is_forward:
                    # En allant dans cette direction, on suit le côté droit
                    direction_map = {
                        'north': [3, 0],  # Côté droit quand on va vers le nord
                        'east': [2, 3],   # Côté droit quand on va vers l'est
                        'south': [1, 2],  # Côté droit quand on va vers le sud
                        'west': [0, 1]    # Côté droit quand on va vers l'ouest
                    }
                else:
                    # En revenant, on visite les sous-cellules restantes (côté gauche)
                    direction_map = {
                        'north': [2, 1],  # Côté gauche quand on allait vers le nord
                        'east': [1, 0],   # Côté gauche quand on allait vers l'est
                        'south': [0, 3],  # Côté gauche quand on allait vers le sud
                        'west': [3, 2]    # Côté gauche quand on allait vers l'ouest
                    }
            elif type == "leaf":
                direction_map = {
                    'north': [1,2,3,0],
                    'east': [0,1,2,3],
                    'south': [3,0,1,2],
                    'west': [2,3,0,1],
                }
            elif type == "node3":
                direction_map = {
                    'north' : [],
                    'east' : [],
                    'south' : [],
                    'west' : [],
                }

            return direction_map.get(direction, [])

        def dfs_coverage(current, parent=None, visited_edges=None):
            """
            DFS modifié pour la couverture STC.
            """
            if visited_edges is None:
                visited_edges = set()

            # Obtenir les voisins dans l'arbre couvrant
            tree_neighbors = list(self.spanning_tree.neighbors(current))

            # Si c'est le nœud de départ, visiter toutes les sous-cellules
            if parent is None:
                # for sub in [0, 1, 2, 3]:
                #     self.coverage_path.append((current, sub))
                if self.spanning_tree.degree(current) <= 1:
                    direction = get_direction(current, tree_neighbors[0])
                    

            # Parcourir les voisins (enfants dans l'arbre)
            for neighbor in tree_neighbors:
                edge = tuple(sorted([current, neighbor]))

                if edge not in visited_edges:
                    visited_edges.add(edge)

                    # Déterminer la direction vers ce voisin
                    direction = get_direction(current, neighbor)

                    if direction:
                        # Si ce n'est pas le nœud de départ, visiter les sous-cellules
                        # du côté droit de cette arête
                        if parent is not None:
                            right_subcells = get_subcells_for_direction(direction, True)
                            for sub in right_subcells:
                                self.coverage_path.append((current, sub))

                        # Récursion vers le voisin
                        dfs_coverage(neighbor, current, visited_edges)

                        # Au retour, visiter les sous-cellules du côté gauche
                        if parent is not None:
                            left_subcells = get_subcells_for_direction(direction, False)
                            for sub in left_subcells:
                                self.coverage_path.append((current, sub))

        # Commencer le DFS depuis le nœud de départ
        dfs_coverage(start_node)

        print(f"Chemin de couverture généré avec {len(self.coverage_path)} pas.")
        print(f"Séquence: {self.coverage_path}")

        return self.coverage_path

    def plan(self, start_point, visualizer=None):
        """
        Exécute le pipeline de planification complet.
        """
        # Étape 2
        self._create_grid()
        if visualizer:
            visualizer.draw_grid(self.cell_polygons.values())
            input("Appuyez sur Entrée pour passer à l'étape 3...")

        # Étape 3
        self._build_graph()
        # Nodes with attributes
        print("Nodes with attributes:", list(self.graph.nodes(data=True)))

        # Adjacency / neighbors
        for node, nbrs in self.graph.adjacency():
            print(node, "->", dict(nbrs)) 
        if visualizer:
            visualizer.draw_graph(self.graph, self.cell_map)
            input("Appuyez sur Entrée pour passer à l'étape 4...")
            
        # Trouver la cellule de départ
        start_cell_idx = self._find_start_cell(start_point)
        print(f"Cellule de départ identifiée: {start_cell_idx}")

        # Étape 4
        self._compute_spanning_tree(start_cell_idx)
        # lists
        print("Nodes:", list(self.spanning_tree.nodes()))
        print("Edges:", list(self.spanning_tree.edges()))
        print("Edges with data:", list(self.spanning_tree.edges(data=True)))

        # DFS order of nodes (preorder)
        print("DFS preorder:", list(nx.dfs_preorder_nodes(self.spanning_tree, source=start_cell_idx)))
        successors = nx.dfs_successors(self.spanning_tree, source=start_cell_idx)

        def print_tree(node, level=0):
            print("  " * level + str(node))
            for child in successors.get(node, []):
                print_tree(child, level + 1)

        print_tree(start_cell_idx)
        if visualizer:
            visualizer.draw_spanning_tree(self.spanning_tree, self.cell_map)
            input("Appuyez sur Entrée pour passser à l'étape 5...")

        # Étape 5
        # self._generate_coverage_path(start_cell_idx)

        # if not self.coverage_path:
        #     print("Le chemin de couverture n'a pas pu être généré.")
        #     return [], None
        # print(f"Chemin de couverture généré: {self.coverage_path}.")
        # # Convertir le chemin de (cell_idx, sub_idx) en polygones
        # path_polygons = [self._get_subcell(c, s) for c, s in self.coverage_path]
        # # print(f"Chemin de couverture généré: {path_polygons}.")
        
        # if visualizer:
        #     anim = visualizer.animate_path(path_polygons)
        #     # On retourne l'animation pour qu'elle ne soit pas garbage-collectée
        #     return path_polygons, anim
            
        # # Les étapes suivantes viendront ici...
        # print("\nPlanification des étapes 1 à 5 terminée.")
        # return path_polygons, None