clear all; close all; clc;
load('sinusoide.mat');

%% Estimateur de Gauss-Markov
Z_1N = Z;
m_1N = zeros(N,1);
R_1N = sigmab^2*eye(N);
% H_1 = [cos(om1*1*Te), cos(om2*1*Te), cos(om3*1*Te)];
% H_2 = [cos(om1*2*Te), cos(om2*2*Te), cos(om3*2*Te)];
% H_3 = [cos(om1*3*Te), cos(om2*3*Te), cos(om3*3*Te)];
H_1N = [cos(om1*(1:N)'*Te), cos(om2*(1:N)'*Te), cos(om3*(1:N)'*Te)];
theta_N_GM = (H_1N'/R_1N*H_1N)\(H_1N'/R_1N)*(Z_1N-m_1N)

%% Estimateur de Gauss-Markov - Alternative
S1 = zeros(3,3);
S2 = zeros(3,1);
for i = 1:N
    H_i = [cos(om1*i*Te), cos(om2*i*Te), cos(om3*i*Te)];
    S1 = S1 + H_i'*H_i;
    S2 = S2 + H_i'*Z(i);
end
theta_N_GM_alternatif = inv(S1)*S2

%% Algorithme des Moindres Carr�s R�cursifs
% Conditions initiales
theta_0 = zeros(3,1);
P_0 = 10*eye(3);
% Initialisation de l'algorithme
theta_n = theta_0;
P_n = P_0;
% Algorithme
vectTheta = zeros(3,N);
vectTheta(:,1) = theta_0;
vectTrP = zeros(1,N);
vectTrP(1) = trace(P_0);
for n = 0:N-1
    m_np1 = 0;
    R_np1 = sigmab^2;
    H_np1 = [cos(om1*(n+1)*Te), cos(om2*(n+1)*Te), cos(om3*(n+1)*Te)];
    % Equations de r�currence
    S_np1 = H_np1*P_n*H_np1' + R_np1;
    K_np1 = P_n*H_np1'/S_np1;
    theta_np1 = theta_n + K_np1*(Z(n+1)-m_np1-H_np1*theta_n);
    P_np1 = P_n - K_np1*H_np1*P_n;
    vectTheta(:,n+1) = theta_np1;
    vectTrP(n+1) = trace(P_np1);
    % Iteration
    theta_n = theta_np1;
    P_n = P_np1;
end
theta_MCR = theta_n

figure; plot(1:N,vectTheta);
title('Evolution de theta_n');
figure; plot(1:N,vectTrP);
title('Evolution de trace(P_n)');
