%% SCRIPT D'ANALYSE DE PERFORMANCE DE LA MISSION DE COUVERTURE
clear; clc; close all;

disp('Lancement de l''analyse de performance de la mission...');

%% 1. Chargement et Extraction des Données
load('mission_data.mat');
disp('Données de la mission chargées.');
% On extrait les données de l'historique
positions = drone_motion_history(:, 1:3);
velocities = drone_motion_history(:, 4:6);
accelerations = drone_motion_history(:, 7:9);
% Le temps total de la mission est le dernier temps d'arrivée
mission_time = time_of_arrival(end);

%% 2. Qualité de la Trajectoire
disp('Calcul des métriques de qualité de la trajectoire...');
% -- Distance Parcourue --
% On calcule la distance entre chaque point consécutif et on somme
distances = vecnorm(diff(positions), 2, 2);
total_distance = sum(distances);

% -- Lissage (Jerk) --
% On calcule le pas de temps (dt) de la simulation
dt = mission_time / size(drone_motion_history, 1);
% On calcule les vecteurs de jerk (différence d'accélération / dt)
jerk_vectors = diff(accelerations) / dt;
% On calcule la magnitude (norme) de chaque vecteur de jerk
jerk_magnitudes = vecnorm(jerk_vectors, 2, 2);
% On calcule le RMS (Root Mean Square) pour avoir une métrique globale
jerk_rms = rms(jerk_magnitudes);

%% 3. Performance de l'Algorithme
% Le temps de calcul a déjà été mesuré et est chargé depuis le fichier.

%% 4. Sécurité et Faisabilité
disp('Calcul des métriques de sécurité...');
% -- Nombre de Collisions --
collision_count = 0;
for i = 1:size(positions, 1)
    pos = positions(i, :); % [North, East, Down]
    for k = 1:length(obstacles)
        % On vérifie si la position (East, North) est dans un polygone d'obstacle
        if inpolygon(pos(2), pos(1), obstacles{k}(:,1), obstacles{k}(:,2))
            collision_count = collision_count + 1;
            break; % Inutile de vérifier les autres obstacles pour ce point
        end
    end
end

%% 5. Efficacité de la Mission
disp('Calcul des métriques d''efficacité...');
%Calcul de l'aire de la zone de mission
% On définit clairement nos axes : X est East, Y est North
zone_polygon = zone_a_surveiller{1};
poly_east_coords = zone_polygon(:, 1); % X
poly_north_coords = zone_polygon(:, 2); % Y

% polyarea(X, Y)
zone_area = polyarea(poly_east_coords, poly_north_coords);

% On fait de même pour les obstacles
obstacles_area = 0;
for i = 1:length(obstacles)
    obstacles_area = obstacles_area + polyarea(obstacles{i}(:,1), obstacles{i}(:,2));
end
target_area = zone_area - obstacles_area;

fprintf('Aire totale de la zone de mission : %.2f m²\n', zone_area);
fprintf('Aire des obstacles à soustraire : %.2f m²\n', obstacles_area);
fprintf('-> Aire cible à couvrir : %.2f m²\n', target_area);

%Calcul de l'aire effectivement couverte
analysis_grid_resolution = 1; % 1 mètre par cellule d'analyse

% Créer les limites de la grille d'analyse en se basant sur la zone
min_east = min(poly_east_coords);
max_east = max(poly_east_coords);
min_north = min(poly_north_coords);
max_north = max(poly_north_coords);

% Créer les vecteurs et la grille dans l'ordre (X, Y) -> (East, North)
east_vect = min_east:analysis_grid_resolution:max_east;
north_vect = min_north:analysis_grid_resolution:max_north;
[east_grid, north_grid] = meshgrid(east_vect, north_vect);
coverage_grid = zeros(size(north_grid));

disp('Calcul de la couverture en cours (cela peut prendre un moment)...');
half_cam_res = camera_resolution / 2;

% Pour chaque point de l'historique du drone...
for i = 1:size(drone_motion_history, 1)
    pos = drone_motion_history(i, :); % [North, East, Down]
    pos_north = pos(1);
    pos_east = pos(2);
    
    % On définit les limites du carré couvert par la caméra
    covered_north_min = pos_north - half_cam_res;
    covered_north_max = pos_north + half_cam_res;
    covered_east_min = pos_east - half_cam_res;
    covered_east_max = pos_east + half_cam_res;
    
    % On "allume" les pixels de la grille qui sont dans ce carré
    indices_to_update = (north_grid >= covered_north_min & north_grid <= covered_north_max & ...
                         east_grid >= covered_east_min & east_grid <= covered_east_max);
    coverage_grid(indices_to_update) = 1;
end

% Maintenant, on s'assure de ne compter que les pixels qui sont à l'intérieur de la zone de mission
% On utilise l'ordre correct pour inpolygon: inpolygon(X_points, Y_points, X_poly, Y_poly)
[in, ~] = inpolygon(east_grid, north_grid, poly_east_coords, poly_north_coords);
coverage_grid(~in) = 0; % On met à 0 les pixels en dehors de la zone

% On fait de même pour les obstacles
for i = 1:length(obstacles)
    [in_obs, ~] = inpolygon(east_grid, north_grid, obstacles{i}(:,1), obstacles{i}(:,2));
    coverage_grid(in_obs) = 0; % On met à 0 les pixels à l'intérieur des obstacles
end

% L'aire couverte est le nombre de pixels allumés multiplié par l'aire de chaque pixel
covered_area = sum(coverage_grid(:)) * (analysis_grid_resolution^2);
fprintf('Aire totale effectivement couverte (sans obstacles) : %.2f m²\n', covered_area);

%Calcul du taux de couverture
coverage_percentage = (covered_area / target_area) * 100;

% -- Consommation de Batterie --
% 1. Définir les paramètres de notre modèle
BATTERY_CAPACITY_WH = 74; % Capacité en Watt-heures (ex: 5000mAh * 14.8V = 74Wh)
AVG_POWER_CONSUMPTION_W = 200; % Consommation moyenne en Watts (estimation pour un quadrotor de cette taille)

% 2. Calculer l'énergie consommée
% mission_time est en secondes, on le convertit en heures ( / 3600)
total_energy_consumed_Wh = AVG_POWER_CONSUMPTION_W * (mission_time / 3600);

% 3. Calculer le pourcentage de batterie utilisé
battery_consumed_percentage = (total_energy_consumed_Wh / BATTERY_CAPACITY_WH) * 100;

%% 6. Affichage des Résultats
fprintf('\n\n--- RAPPORT DE PERFORMANCE DE LA MISSION ---\n\n');

fprintf('** Qualité de la Trajectoire **\n');
fprintf('   - Distance totale parcourue : %.2f mètres\n', total_distance);
fprintf('   - Temps total de la mission : %.2f secondes\n', mission_time);
fprintf('   - Lissage du vol (Jerk RMS) : %.4f m/s³ (plus c''est bas, plus c''est fluide)\n\n', jerk_rms);

fprintf('** Performance de l''Algorithme **\n');
fprintf('   - Temps de calcul du chemin : %.4f secondes\n\n', computation_time);

fprintf('** Sécurité et Faisabilité **\n');
fprintf('   - Nombre de points en collision : %d\n', collision_count);
if collision_count > 0
    fprintf('   - VERDICT SÉCURITÉ : ÉCHEC (Collision détectée)\n\n');
else
    fprintf('   - VERDICT SÉCURITÉ : SUCCÈS (Aucune collision)\n\n');
end

fprintf('** Efficacité de la Mission **\n');
fprintf('   - Taux de couverture de la zone : %.2f %%\n', coverage_percentage); % Décommentez quand vous collez votre code
fprintf('   - Batterie consommée (estimation) : %.2f %%\n', battery_consumed_percentage);
if battery_consumed_percentage > 100
    fprintf('   - AVERTISSEMENT : La mission requiert plus de 100%% de la batterie !\n');
end

fprintf('\n--- FIN DU RAPPORT ---\n');

%% 7. Visualisation de la grille de couverture
figure;
hold on;
title('Visualisation de la Couverture Effective');
% On affiche la grille de couverture. surf(X, Y, Z)
surf(east_grid, north_grid, coverage_grid, 'EdgeColor', 'none');
% On dessine les polygones par-dessus. plot(X, Y)
plot(poly_east_coords([1:end,1]), poly_north_coords([1:end,1]), 'k', 'LineWidth', 2);
for i = 1:length(obstacles)
    fill(obstacles{i}(:,1), obstacles{i}(:,2), 'r');
end
xlabel('East (m)');
ylabel('North (m)');
view(0, 90);
axis equal;
legend('Zone Couverte');
disp('Visualisation de la couverture générée.');