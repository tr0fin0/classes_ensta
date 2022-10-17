clear all; close all; clc;

load('filtreKalman.mat'); % Contient Z, X, N, T, sigma_p, sigma_v, sigma_z

F = [1,T,0,0;0,1,0,0;0,0,1,T;0,0,0,1];
H = [1,0,0,0;0,0,1,0];
W = T*[sigma_p^2,0,0,0;0,sigma_v^2,0,0;0,0,sigma_p^2,0;0,0,0,sigma_v^2];
V = [sigma_z^2,0;0,sigma_z^2];


% Filtre de Kalman
X_est = zeros(4,N);
varP = zeros(1,N);
% Initialisation du filtre
X_est(:,1) = [0;0;0;0]; % Inutile, mais pour marquer le coup de l'initialisation
x_k = [0;0;0;0];
P_k = 1000*eye(4);
varP(1,1) = trace(P_k);
for k = 1:N-1
    % Pr�diction
    x_kp_k = F*x_k;
    P_kp_k = F*P_k*F'+W;
    % Calcul du gain de Kalman
    K_kp = P_kp_k*H'*inv(H*P_kp_k*H'+V);
    % Mise � jour / Correction
    x_kp = x_kp_k + K_kp*(Z(:,k+1)-H*x_kp_k);
    P_kp = (eye(4)-K_kp*H)*P_kp_k;
    % Sauvegarde des donn�es d'int�r�t
    X_est(:,k+1) = x_kp;
    varP(1,k+1) = trace(P_kp);
    % En vue de l'it�ration suivante
    x_k = x_kp;
    P_k = P_kp;
end

% Affichage des r�sultats
figure;
subplot(221); plot(1:N,[X(1,:);X_est(1,:)]); ylabel('Position selon x'); legend('R�el','Estimation');
subplot(222); plot(T:T:T*N,[X(2,:);X_est(2,:)]); ylabel('Vitesse selon x'); legend('R�el','Estimation');
subplot(223); plot(T:T:T*N,[X(3,:);X_est(3,:)]); ylabel('Position selon y'); legend('R�el','Estimation');
subplot(224); plot(T:T:T*N,[X(4,:);X_est(4,:)]); ylabel('Vitesse selon y'); legend('R�el','Estimation');

% Vecteur d'erreur
erreur = zeros(1,N);
for k = 1:N
    erreur(k) = sum((X(:,k)-X_est(:,k)).^2);
end

figure;
plot(1:N,[erreur;varP]);
