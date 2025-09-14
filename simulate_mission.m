%% Étape 0: Initialisation de l'environnement
clear; clc; close all;

disp('Initialisation de la simulation de mission de drone...');

% Assurez-vous que MATLAB utilise le bon interpréteur Python
% (normalement, on le configure une seule fois, mais c'est bien de le vérifier)
pyenv; 

%% Étape 1: Définition des paramètres de la mission (comme en Python)
disp('Définition des paramètres de la mission...');
FLIGHT_ALTITUDE_M = 40;     % Altitude de vol en mètres(40)
CAMERA_FOV_DEGREES = 30;    % Angle de vue de la caméra en degrés(60)
OVERLAP_PERCENTAGE = 0;   % Pourcentage de chevauchement entre les passes (0-99)
START_POINT = [100, 100]; % Coordonnées (East, North)

% Définir la zone de surveillance. Utiliser des "cell arrays" pour les polygones.
%zone_a_surveiller = {[50 50; 450 50; 450 350; 50 350]};

% Un polygone concave en forme de "U" couché.
% Les sommets sont listés dans l'ordre pour former le polygone.
zone_a_surveiller = {[
    50, 50;   % 1. Départ en bas à gauche
    450, 50;  % 2. Vers la droite
    450, 350; % 3. Vers le haut
    300, 350; % 4. Rentre vers la gauche
    300, 150; % 5. Descend (crée la concavité)
    200, 150; % 6. Vers la gauche
    200, 350; % 7. Remonte
    50, 350   % 8. Retour au début (haut gauche)
]};

% Définir les obstacles
% obstacles = {
%     [150 150; 250 150; 250 250; 150 250]
% };
obstacles = {[100 100; 150 100; 150 150; 100 150]};

%% Étape 2: Appel du planificateur Python pour obtenir les waypoints
disp('Appel du script Python pour la planification de la trajectoire...');

% Insérer le répertoire courant dans le chemin de Python pour qu'il trouve le module
if count(py.sys.path,'.') == 0
    insert(py.sys.path,int32(0),'.');
end

% Importer le module Python
planner_module = py.importlib.import_module('stc_path_planner');

% Forcer le rechargement du module
py.importlib.reload(planner_module);

% Appeler la fonction 'planifier_mission'
% Note: MATLAB convertit automatiquement ses types de données
tic;
py_mission_data = planner_module.planifier_mission(zone_a_surveiller{1}, FLIGHT_ALTITUDE_M, CAMERA_FOV_DEGREES, obstacles, START_POINT, OVERLAP_PERCENTAGE);
computation_time = toc;
fprintf('Temps de calcul de l''algorithme Python : %.4f secondes.\n', computation_time);

if isempty(py_mission_data)
    error('Le script Python n a retourné aucune donnée. Arrêt.');
end

% On extrait les données du dictionnaire
py_waypoints = py_mission_data{'waypoints'};
camera_resolution = double(py_mission_data{'resolution_m'}); % On récupère la résolution !
fprintf('Résolution de la caméra reçue : %.2f m\n', camera_resolution);
% Conversion des waypoints de Python vers une matrice MATLAB
waypoints_count = int64(py_waypoints.length);
if waypoints_count == 0
    error('Le script Python n a retourné aucun waypoint. Arrêt de la simulation.');
end
waypoints_mat = zeros(waypoints_count, 2);
for i = 1:waypoints_count
    waypoints_mat(i, :) = cell2mat(cell(py_waypoints{i}));
end

% Ajouter l'altitude pour créer une trajectoire 3D
% On assemble la matrice de waypoints 3D en respectant le système NED (North-East-Down)
% 1. On prend la 2ème colonne de Python (North) comme 1ère coordonnée.
% 2. On prend la 1ère colonne de Python (East) comme 2ème coordonnée.
% 3. On utilise l'altitude NÉGATIVE pour la 3ème coordonnée (Down).
waypoints_3d = [waypoints_mat(:,2), waypoints_mat(:,1), -ones(waypoints_count, 1) * FLIGHT_ALTITUDE_M];

fprintf('Trajectoire reçue avec %d waypoints.\n', waypoints_count);

%% Étape 3: Création de l'environnement de simulation UAV Toolbox
disp('Création de l environnement de simulation 3D...');

scenario = uavScenario('UpdateRate', 10, 'ReferenceLocation', [0 0 0]);
drone = uavPlatform('UAV', scenario, 'ReferenceFrame', 'NED');

% On rend le drone visible en utilisant la syntaxe complète pour R2021a
% updateMesh(plateforme, type, {échelle}, [R G B], position_relative, orientation_relative);
% On fournit explicitement les valeurs par défaut pour la position et l'orientation.
updateMesh(drone, 'quadrotor', {20}, [0.5 0.5 0.5], [0 0 0], [1 0 0 0]); 

% 1. Définir une vitesse de vol pour le drone (en mètres/seconde)
DRONE_SPEED_MS = 40.0; % 10 m/s = 36 km/h
fprintf('Calcul des temps d''arrivée pour une vitesse de %d m/s.\n', DRONE_SPEED_MS);

% 2. Préparer le vecteur pour les temps d'arrivée
num_waypoints = size(waypoints_3d, 1);
time_of_arrival = zeros(num_waypoints, 1);

% 3. Calculer le temps cumulé pour atteindre chaque waypoint
% Le premier waypoint est au temps 0
for i = 2:num_waypoints
    % Distance entre le waypoint actuel et le précédent
    distance_segment = norm(waypoints_3d(i, :) - waypoints_3d(i-1, :));
    
    % Temps pour parcourir ce segment
    time_segment = distance_segment / DRONE_SPEED_MS;
    
    % Le temps d'arrivée est le temps d'arrivée précédent + le temps du segment
    time_of_arrival(i) = time_of_arrival(i-1) + time_segment;
end

% On fournit maintenant 'waypoints_3d' ET 'time_of_arrival'
trajectory = waypointTrajectory(waypoints_3d,'TimeOfArrival', time_of_arrival,'SampleRate', scenario.UpdateRate,'AutoPitch', true);

%% Étape 4: Visualisation de la simulation
disp('Lancement de la simulation...');

% Créer une vue 3D du scénario et garder le handle de l'axe (ax)
fig = figure; % Ouvre une nouvelle fenêtre de figure
ax = show3D(scenario);
hold on;

% Dessiner la zone de mission et les obstacles 
zone_plot = zone_a_surveiller{1};
plot3(ax, zone_a_surveiller{1}([1:end,1],1), zone_a_surveiller{1}([1:end,1],2), zeros(size(zone_a_surveiller{1},1)+1,1), 'b', 'LineWidth', 2);
for i = 1:length(obstacles)
    obs_plot = obstacles{i};
    fill3(ax, obstacles{i}(:,1), obstacles{i}(:,2), zeros(size(obstacles{i},1),1), 'r');
end
% On inverse aussi les coordonnées pour la ligne verte pour qu'elle corresponde au drone
plot3(ax, waypoints_3d(:,2), waypoints_3d(:,1), -waypoints_3d(:,3), 'g--');
title('Simulation de la mission de surveillance');
hold off;

% On configure la caméra une seule fois, avant la boucle de simulation.
% view(0, 90) est la commande standard pour une vue de dessus parfaite (azimut=0, élévation=90).
view(0, 90);
% axis equal empêche la distorsion des formes (un carré ressemblera à un carré).
axis equal;

% 1. Définir le nom du fichier de sortie
outputVideoFilename = 'mission_simulation.mp4';
% 2. Créer un objet VideoWriter avec le profil MPEG-4
videoWriter = VideoWriter(outputVideoFilename, 'MPEG-4');
% 3. Ouvrir le fichier pour l'écriture
open(videoWriter);
fprintf('Enregistrement de la simulation dans %s\n', outputVideoFilename);

% 1. Calculer le nombre total d'itérations que la boucle va faire
total_simulation_time = time_of_arrival(end);
num_simulation_steps = ceil(total_simulation_time * scenario.UpdateRate);

% 2. Préallouer la mémoire pour l'historique complet du mouvement
% On crée une matrice de zéros de la taille finale exacte.
drone_motion_history = zeros(num_simulation_steps, 16);
fprintf('Préallocation de la mémoire pour %d points d''historique.\n', num_simulation_steps);

% Boucle de simulation
setup(scenario);
i = 1;
num_waypoints = size(waypoints_3d, 1);
while ~isDone(trajectory)
    % 1. Lire la position actuelle sur la trajectoire
    [pos, orient, vel, acc, angvel] = trajectory();
    
    % 2. Convertir l'objet quaternion 'orient' en un vecteur numérique [w x y z]
    orientationVector = compact(orient);
    
    % 3. Assembler le vecteur de 16 éléments dans l'ordre exact spécifié par la documentation
    motionVector = [pos, vel, acc, orientationVector, angvel];
    
    % On sauvegarde le vecteur de mouvement complet
    drone_motion_history(i, :) = motionVector;
    
    % 4. Passer ce vecteur de 16 éléments à la fonction move
    move(drone, motionVector);
    
    % Mettre à jour les capteurs et l'environnement
    advance(scenario);
    
    % Mettre à jour la visualisation 3D
    show3D(scenario, "FastUpdate", true, "Parent", ax);
    
    % CONTRÔLER LA CAMÉRA DE LA FENÊTRE 3D
    % La caméra regardera toujours le drone
    %camtarget(ax, pos);
    % La caméra sera positionnée légèrement derrière et au-dessus du drone
    % pour une vue "poursuite"
    %camera_offset = [-15, 0, -5]; % Décalage en (x, y, z) par rapport au drone
    %campos(ax, pos + camera_offset);
    
    % On calcule les 4 coins du carré de couverture au sol (z=0)
    half_res = camera_resolution / 4;
    % pos(1)=North, pos(2)=East
    patch_y = [pos(1)-half_res, pos(1)-half_res, pos(1)+half_res, pos(1)+half_res];
    patch_x = [pos(2)-half_res, pos(2)+half_res, pos(2)+half_res, pos(2)-half_res];
    % On dessine le patch avec une couleur verte semi-transparente
    patch(patch_x, patch_y, 'g', 'FaceAlpha', 0.05, 'EdgeColor', 'none');
    
    drawnow limitrate;
    
    % On capture l'intégralité de la fenêtre de la figure 'fig'
    frame = getframe(fig);
    % On écrit cette image dans le fichier vidéo
    writeVideo(videoWriter, frame);

    if mod(i, 10) == 0 
        % Obtenir le temps actuel de la simulation
        currentTime = scenario.CurrentTime;
        
        % Trouver l'index du premier temps d'arrivée qui est supérieur ou égal au temps actuel
        % C'est le waypoint vers lequel le drone se dirige
        nextWaypointIndex = find(time_of_arrival >= currentTime, 1, 'first');
        
        % Gérer le cas où on a dépassé le dernier waypoint
        if isempty(nextWaypointIndex)
            nextWaypointIndex = num_waypoints;
        end
        
        % Afficher le message de progression correct
        fprintf('Simulation en cours... Vers le waypoint %d / %d (Temps: %.1fs)\n', nextWaypointIndex, num_waypoints, currentTime);
    end
    i=i+1;
end

% Il est possible que la simulation se termine avec quelques étapes de moins que prévu.
% On supprime donc les lignes de zéros en trop qui n'ont pas été remplies.
drone_motion_history = drone_motion_history(1:i, :);

% On ferme le fichier vidéo. C'est une étape cruciale.
close(videoWriter);
fprintf('Vidéo enregistrée avec succès.\n');

disp('Sauvegarde des données de la mission pour l''analyse...');
save('mission_data.mat', 'drone_motion_history', 'zone_a_surveiller', 'camera_resolution', 'obstacles', 'computation_time', 'time_of_arrival');
fprintf('Données sauvegardées dans mission_data.mat\n');

disp('Simulation terminée.');